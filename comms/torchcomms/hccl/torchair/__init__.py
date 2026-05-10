# torchair convertor registration for the hccl backend.
#
# Importing this module triggers FX → GE node conversion registrations for
# any torchcomms collective op that is exposed as torch.ops.torchcomms.*.
# Requires torchair (provided by torch_npu.dynamo.torchair) and an activated
# torchcomms functional/ path (TORCHCOMMS_PATCH_FOR_COMPILE=1; on torch < 2.12
# also TORCHCOMMS_COMPILE_IGNORE_PYTORCH_VERSION_REQUIREMENT=1).

try:
    import torch_npu.dynamo.torchair  # noqa: F401
except ImportError as e:
    raise ImportError(
        "torchcomms.hccl.torchair requires torch_npu.dynamo.torchair; "
        "install torch_npu and source the cann set_env.sh first."
    ) from e

from . import converters  # noqa: F401
