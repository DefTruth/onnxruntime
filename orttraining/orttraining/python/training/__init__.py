# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from onnxruntime.capi._pybind_state import TrainingParameters
from onnxruntime.capi._pybind_state import PropagateCastOpsStrategy
from onnxruntime.capi.training.training_session import TrainingSession

from .orttrainer_options import ORTTrainerOptions
from .orttrainer import ORTTrainer, TrainStepInfo
from . import amp, checkpoint, optim, model_desc_validation

# from onnxruntime.training.ortmodule._fallback import ORTModuleInitException

try:
    from .ortmodule import ORTModule
except ImportError:
    # Not a ORTModule training package
    pass
# except ORTModuleInitException:
#     # Not a ORTModule training package
#     pass
# ############################################################################################
except Exception as e:
    print('orttraining/__init__.py: except try 1 exception')
    try:
        print('orttraining/__init__.py: try 2')
        from onnxruntime.training.ortmodule._fallback import ORTModuleInitException
        if isinstance(e, ORTModuleInitException):
            print('orttraining/__init__.py: try 2 import ok')
            # ORTModule is present but not ready to run
            # That is OK when this is not a ORTModule training package
            pass
    except Exception as ee:
        print(f'orttraining/__init__.py: except try 2 {ee}')
        # Not a ORTModule training package
        pass