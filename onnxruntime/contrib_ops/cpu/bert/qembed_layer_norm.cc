// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "qembed_layer_norm.h"

#include <cmath>

#include "contrib_ops/cpu/qlinear_lookup_table.h"
#include "core/framework/op_kernel.h"
#include "core/providers/common.h"
#include "embed_layer_norm_helper.h"

namespace onnxruntime {
namespace contrib {

namespace {

constexpr size_t kLookupTableSize = 256;

// TODO(kreeger): Drop this when ComputeInternal() is using a lookup table.
template <typename T>
inline float Dequantize(T value, float scale, T zero_point) {
  return static_cast<float>(static_cast<int32_t>(value) - zero_point) * scale;
}

template <typename T, typename T2>
Status ComputeInternal(OpKernelContext* context, float epsilon) {
  // TODO(kreeger): Let's move these values to a constexpr becuase they are dupe'd throughout this file!
  const Tensor* input_ids = context->Input<Tensor>(0);
  const Tensor* segment_ids = context->Input<Tensor>(1);  // optional. nullptr if it's distill-bert
  const Tensor* word_embedding = context->Input<Tensor>(2);
  const Tensor* position_embedding = context->Input<Tensor>(3);
  const Tensor* segment_embedding = context->Input<Tensor>(4);  // optional. nullptr if it's distill-bert
  const Tensor* gamma = context->Input<Tensor>(5);
  const Tensor* beta = context->Input<Tensor>(6);
  const Tensor* mask = context->Input<Tensor>(7);  // optional. nullptr if not provided
  const Tensor* word_embedding_scale = context->Input<Tensor>(8);
  const Tensor* position_embedding_scale = context->Input<Tensor>(9);
  const Tensor* segment_embedding_scale = context->Input<Tensor>(10);
  const Tensor* gamma_scale = context->Input<Tensor>(11);
  const Tensor* beta_scale = context->Input<Tensor>(12);
  const Tensor* word_embedding_zero_point = context->Input<Tensor>(13);
  const Tensor* position_embedding_zero_point = context->Input<Tensor>(14);
  const Tensor* segment_embedding_zero_point = context->Input<Tensor>(15);
  const Tensor* gamma_zero_point = context->Input<Tensor>(16);
  const Tensor* beta_zero_point = context->Input<Tensor>(17);

  const auto& input_dims = input_ids->Shape().GetDims();
  int batch_size = static_cast<int>(input_dims[0]);
  int sequence_length = static_cast<int>(input_dims[1]);
  int64_t hidden_size = word_embedding->Shape()[1];

  // Request outputs:
  TensorShape output_shape({batch_size, sequence_length, hidden_size});
  Tensor* output = context->Output(0, output_shape);

  TensorShape mask_index_shape({batch_size});
  Tensor* mask_index = context->Output(1, mask_index_shape);
  bool has_segment_embedding = segment_ids != nullptr;

  const int32_t* input_ids_data = input_ids->template Data<int32_t>();
  const int32_t* segment_ids_data =
      has_segment_embedding ? segment_ids->template Data<int32_t>() : nullptr;

  int word_embedding_length = static_cast<int>(word_embedding->Shape()[0]);
  int position_embedding_length = static_cast<int>(position_embedding->Shape()[0]);
  int segment_embedding_length =
      has_segment_embedding ? static_cast<int>(segment_embedding->Shape()[0]) : 0;

  // Grab quantization values:
  float word_embedding_scale_data = *(word_embedding_scale->template Data<float>());
  T2 word_embedding_zero_point_data = *(word_embedding_zero_point->template Data<T2>());

  float position_embedding_scale_data = *(position_embedding_scale->template Data<float>());
  T2 position_embedding_zero_point_data = *(position_embedding_zero_point->template Data<T2>());

  float segment_embedding_scale_data =
      has_segment_embedding ? *(segment_embedding_scale->template Data<float>()) : 0.0f;
  T2 segment_embedding_zero_point_data =
      has_segment_embedding ? *(segment_embedding_zero_point->template Data<T2>()) : 0;

  float gamma_scale_data = *(gamma_scale->template Data<float>());
  T2 gamma_zero_point_data = *(gamma_zero_point->template Data<T2>());

  float beta_scale_data = *(beta_scale->template Data<float>());
  T2 beta_zero_point_data = *(beta_zero_point->template Data<T2>());

  // Grab pointers to buffers each Tensor represents:
  const T2* word_embedding_data = word_embedding->template Data<T2>();
  const T2* position_embedding_data = position_embedding->template Data<T2>();
  const T2* segment_embedding_data =
      has_segment_embedding ? segment_embedding->template Data<T2>() : nullptr;
  const T2* gamma_data = gamma->template Data<T2>();
  const T2* beta_data = beta->template Data<T2>();

  T* output_data = output->template MutableData<T>();

  // Perform the Op:
  {
    std::atomic_bool failed{false};

    // TODO: Profile and tune this batch parallel execution based on input size.
    // More info: https://github.com/microsoft/onnxruntime/pull/8124/files#r656629895
    int n = batch_size * sequence_length;
    concurrency::ThreadPool::TryBatchParallelFor(
        context->GetOperatorThreadPool(), n, [=, &failed](ptrdiff_t index) {
          int word_col_index = input_ids_data[index];
          if (word_col_index < 0 || word_col_index >= word_embedding_length) {
            failed.store(true, std::memory_order_release);
            return;
          }
          int position_col_index = index % sequence_length;
          if (position_col_index >= position_embedding_length) {
            failed.store(true, std::memory_order_release);
            return;
          }
          int segment_col_index = 0;
          if (nullptr != segment_ids_data) {
            segment_col_index = segment_ids_data[index];
            if (segment_col_index < 0 || segment_col_index >= segment_embedding_length) {
              failed.store(true, std::memory_order_release);
              return;
            }
          }

          // Grab inputs for the embeddings for the current batch index:
          const T2* input_word_embedding = word_embedding_data + word_col_index * hidden_size;
          const T2* input_position_embedding =
              position_embedding_data + position_col_index * hidden_size;
          const T2* input_segment_embedding = nullptr;
          if (segment_embedding_data != nullptr) {
            input_segment_embedding = segment_embedding_data + segment_col_index * hidden_size;
          }

          T* output = output_data + (index * hidden_size);

          // TODO - let's do all this at int8/uint8 - it can be done? 

          T2 sum2 = static_cast<T2>(0);

          T sum = static_cast<T>(0);  // TODO - drop this!
          for (int i = 0; i < hidden_size; ++i) {
            // TODO(kreeger): Use a table query to improve performance:
            //T subtotal = Dequantize<T2>(input_word_embedding[i],
            //                            word_embedding_scale_data,
            //                            word_embedding_zero_point_data) +
            //             Dequantize<T2>(input_position_embedding[i],
            //                            position_embedding_scale_data,
            //                            position_embedding_zero_point_data);
            T subtotal = 
            if (segment_embedding_data != nullptr) {
              subtotal += Dequantize<T2>(input_segment_embedding[i],
                                         segment_embedding_scale_data,
                                         segment_embedding_zero_point_data);
            }

            // TODO - convert to float here?
            output[i] = subtotal;
            sum += subtotal;
          }

          T mean = sum / hidden_size;
          sum = 0;

          for (int i = 0; i < hidden_size; i++) {
            T a = output[i] - mean;
            output[i] = a;
            sum += a * a;
          }

          T e = sqrt(sum / hidden_size + epsilon);
          for (int i = 0; i < hidden_size; i++) {
            // TODO(kreeger): Consider keeping these as int8 or use PrePack()!
            T cur_gamma = Dequantize<T2>(gamma_data[i], gamma_scale_data, gamma_zero_point_data);
            T cur_beta = Dequantize<T2>(beta_data[i], beta_scale_data, beta_zero_point_data);
            output[i] = output[i] / e * cur_gamma + cur_beta;
          }
        },
        0);

    if (failed.load(std::memory_order_acquire)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "input index out of range");
    }
  }

  // Calculate mask
  if (nullptr != mask) {
    const int32_t* mask_data = mask->template Data<int32_t>();
    int32_t* mask_index_data = mask_index->template MutableData<int32_t>();
    for (int b = 0; b < batch_size; b++) {
      int32_t cur_sum = 0;
      const int32_t* cur_mask_data = mask_data + (static_cast<int64_t>(b) * sequence_length);
      for (int s = 0; s < sequence_length; ++s) {
        if (cur_mask_data[s] == 1) {
          cur_sum += cur_mask_data[s];
        }
      }
      mask_index_data[b] = cur_sum;
    }
  } else {
    memset(mask_index->template MutableData<int32_t>(), 0, batch_size * sizeof(int32_t));
  }
  return Status::OK();
}

Status CheckQuantizedInputs(OpKernelContext* context, bool* is_signed_inputs) {
  const Tensor* word_embedding_scale_tensor = context->Input<Tensor>(8);
  const Tensor* position_embedding_scale_tensor = context->Input<Tensor>(9);
  const Tensor* segment_embedding_scale_tensor = context->Input<Tensor>(10);
  const Tensor* gamma_scale_tensor = context->Input<Tensor>(11);
  const Tensor* beta_scale_tensor = context->Input<Tensor>(12);
  const Tensor* word_embedding_zero_point_tensor = context->Input<Tensor>(13);
  const Tensor* position_embedding_zero_point_tensor = context->Input<Tensor>(14);
  const Tensor* segment_embedding_zero_point_tensor = context->Input<Tensor>(15);
  const Tensor* gamma_zero_point_tensor = context->Input<Tensor>(16);
  const Tensor* beta_zero_point_tensor = context->Input<Tensor>(17);

  bool word_embedding_is_signed_inputs = word_embedding_scale_tensor->IsDataType<int8_t>();
  bool has_segment_embedding = context->Input<Tensor>(1) != nullptr;

  if (!IsScalarOr1ElementVector(word_embedding_scale_tensor)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Word embedding scale must be a scalar or 1D tensor of size 1");
  }

  if (!IsScalarOr1ElementVector(position_embedding_scale_tensor) &&
      position_embedding_scale_tensor->IsDataType<int8_t>() == word_embedding_is_signed_inputs) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Position embedding scale must be a scalar or 1D tensor of size 1");
  }

  if (has_segment_embedding && !IsScalarOr1ElementVector(segment_embedding_scale_tensor) &&
      segment_embedding_scale_tensor->IsDataType<int8_t>() == word_embedding_is_signed_inputs) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Segment embedding scale must be a scalar or 1D tensor of size 1");
  }

  if (!IsScalarOr1ElementVector(gamma_scale_tensor) &&
      gamma_scale_tensor->IsDataType<int8_t>() == word_embedding_is_signed_inputs) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Gamma scale must be a scalar or 1D tensor of size 1");
  }

  if (!IsScalarOr1ElementVector(beta_scale_tensor) &&
      beta_scale_tensor->IsDataType<int8_t>() == word_embedding_is_signed_inputs) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Beta scale must be a scalar or 1D tensor of size 1");
  }

  if (!IsScalarOr1ElementVector(word_embedding_zero_point_tensor) &&
      word_embedding_zero_point_tensor->IsDataType<int8_t>() == word_embedding_is_signed_inputs) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Word embedding zero point must be a scalar or 1D tensor of size 1");
  }

  if (!IsScalarOr1ElementVector(position_embedding_zero_point_tensor) &&
      position_embedding_zero_point_tensor->IsDataType<int8_t>() == word_embedding_is_signed_inputs) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Position embedding zero point must be a scalar or 1D tensor of size 1");
  }

  if (has_segment_embedding && !IsScalarOr1ElementVector(segment_embedding_zero_point_tensor) &&
      segment_embedding_zero_point_tensor->IsDataType<int8_t>() == word_embedding_is_signed_inputs) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Segment embedding zero point must be a scalar or 1D tensor of size 1");
  }

  if (!IsScalarOr1ElementVector(gamma_zero_point_tensor) &&
      gamma_zero_point_tensor->IsDataType<int8_t>() == word_embedding_is_signed_inputs) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Gamma zero point must be a scalar or 1D tensor of size 1");
  }

  if (!IsScalarOr1ElementVector(beta_zero_point_tensor) &&
      beta_zero_point_tensor->IsDataType<int8_t>() == word_embedding_is_signed_inputs) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Beta zero point must be a scalar or 1D tensor of size 1");
  }

  *is_signed_inputs = word_embedding_is_signed_inputs;
  return Status::OK();
}

Status PopulateConstantLookupTable(const OpKernelInfo& info,
                                   std::vector<uint8_t>& table) {
  const Tensor* word_embedding_scale_tensor = nullptr;
  ORT_RETURN_IF_NOT(info.TryGetConstantInput(/*input_index=*/8, &word_embedding_scale_tensor),
                    "Could not load constant tensor at index %d", 8);

  const Tensor* position_embedding_scale_tensor = nullptr;
  ORT_RETURN_IF_NOT(info.TryGetConstantInput(/*input_index=*/9, &position_embedding_scale_tensor),
                    "Could not load constant tensor at index %d", 9);

  const Tensor* word_embedding_zero_point_tensor = nullptr;
  ORT_RETURN_IF_NOT(info.TryGetConstantInput(/*input_index=*/13, &word_embedding_zero_point_tensor),
                    "Could not load constant tensor at index %d", 13);

  const Tensor* position_embedding_zero_point_tensor = nullptr;
  ORT_RETURN_IF_NOT(info.TryGetConstantInput(/*input_index=*/14, &position_embedding_zero_point_tensor),
                    "Could not load constant tensor at index %d", 14);

  // TODO(kreeger): Check both ZPs for uint8/int8?
  bool is_signed_int8 = word_embedding_zero_point_tensor->IsDataType<int8_t>();

  // TODO(kreeger): check for int8/uint8/float for these tensors?!

  // 256 elements are cached in the table:
  table.resize(kLookupTableSize);

  // Identity function (NOTE: useless comment for now)
  // NOTE: This is the signature for LookupTableScalarTransfomer - might need the Array Transformer one!
  const auto identity_float = [](float v) -> float { return v; };

  //
  //
  // NOTE: Left off right here - think about this logic here and transcribe it to int8?!
  //
  //  
            //T subtotal = Dequantize<T2>(input_word_embedding[i],
            //                            word_embedding_scale_data,
            //                            word_embedding_zero_point_data) +
            //             Dequantize<T2>(input_position_embedding[i],
            //                            position_embedding_scale_data,
            //                            position_embedding_zero_point_data);


  // what is tensor-y here?
  if (is_signed_int8) {
    QlinearBuildLookupTable<int8_t>(table.data(),
                                    word_embedding_scale_tensor,
                                    word_embedding_zero_point_tensor,
                                    position_embedding_scale_tensor,
                                    position_embedding_zero_point_tensor,
                                    identity_float);
  } else {
    QlinearBuildLookupTable<uint8_t>(table.data(),
                                    word_embedding_scale_tensor,
                                    word_embedding_zero_point_tensor,
                                    position_embedding_scale_tensor,
                                    position_embedding_zero_point_tensor,
                                    identity_float);
  }

  return Status::OK();
}

}  // namespace

// This op is internal-only, so register outside of onnx:
#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      QEmbedLayerNormalization,                                   \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCpuExecutionProvider,                                      \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      QEmbedLayerNorm<T>);

REGISTER_KERNEL_TYPED(float)

template <typename T>
QEmbedLayerNorm<T>::QEmbedLayerNorm(const OpKernelInfo& op_kernel_info)
    : EmbedLayerNormBase(op_kernel_info) {
  PopulateConstantLookupTable(op_kernel_info, word_position_embedding_lookup_table_);
}

template <typename T>
Status QEmbedLayerNorm<T>::Compute(OpKernelContext* context) const {
  ORT_RETURN_IF_ERROR(embed_layer_norm::CheckInputs(context));

  bool is_signed_inputs = false;
  ORT_RETURN_IF_ERROR(CheckQuantizedInputs(context, &is_signed_inputs));

  if (is_signed_inputs) {
    return ComputeInternal<T, int8_t>(context, epsilon());
  } else {
    return ComputeInternal<T, uint8_t>(context, epsilon());
  }
}

}  // namespace contrib
}  // namespace onnxruntime