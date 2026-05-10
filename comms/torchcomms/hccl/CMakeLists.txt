# Copyright (c) 2026 Shan Shan.
# Extension: torchcomms._comms_hccl
file(GLOB TORCHCOMMS_HCCL_SOURCES "comms/torchcomms/hccl/*.cpp")
file(GLOB TORCHCOMMS_NPU_API_SOURCES
    "comms/torchcomms/device/npu/*.cpp"
)

# Locate CANN: prefer ASCEND_HOME_PATH (cann 9.0+ env var), fall back to
# ASCEND_TOOLKIT_HOME (cann 8.x), or TORCHCOMMS_CANN_HOME for explicit override.
if(DEFINED TORCHCOMMS_CANN_HOME)
    set(CANN_HOME "${TORCHCOMMS_CANN_HOME}")
elseif(DEFINED ENV{ASCEND_HOME_PATH})
    set(CANN_HOME "$ENV{ASCEND_HOME_PATH}")
elseif(DEFINED ENV{ASCEND_TOOLKIT_HOME})
    set(CANN_HOME "$ENV{ASCEND_TOOLKIT_HOME}")
else()
    message(FATAL_ERROR
        "USE_HCCL=ON but no CANN home found. "
        "Set ASCEND_HOME_PATH (source set_env.sh) or "
        "-DTORCHCOMMS_CANN_HOME=/path/to/cann")
endif()
message(STATUS "  CANN_HOME : ${CANN_HOME}")

set(HCCL_INCLUDE "${CANN_HOME}/include")
set(HCCL_LIB_DIR "${CANN_HOME}/lib64")
set(HCCL_SHARED_LIB "${HCCL_LIB_DIR}/libhccl.so")
set(ACL_SHARED_LIB "${HCCL_LIB_DIR}/libascendcl.so")

# Locate torch_npu (for c10_npu::NPUGuard / getCurrentNPUStream).
# This is the only place we link torch_npu.
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
        "import torch_npu, os; print(os.path.dirname(torch_npu.__file__))"
    OUTPUT_VARIABLE TORCH_NPU_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE TORCH_NPU_RESOLVE_RC
)
if(NOT TORCH_NPU_RESOLVE_RC EQUAL 0 OR "${TORCH_NPU_DIR}" STREQUAL "")
    message(FATAL_ERROR
        "USE_HCCL=ON but cannot import torch_npu. "
        "Make sure PYTHONPATH points to a working torch_npu install.")
endif()
message(STATUS "  TORCH_NPU_DIR : ${TORCH_NPU_DIR}")
set(TORCH_NPU_INCLUDE "${TORCH_NPU_DIR}/include")
set(TORCH_NPU_LIB "${TORCH_NPU_DIR}/lib/libtorch_npu.so")

add_library(torchcomms_comms_hccl MODULE
    ${TORCHCOMMS_HCCL_SOURCES}
    ${TORCHCOMMS_NPU_API_SOURCES}
)
set_target_properties(torchcomms_comms_hccl PROPERTIES
    PREFIX ""
    OUTPUT_NAME "_comms_hccl"
    SUFFIX ".${Python3_SOABI}.so"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/comms/torchcomms"
)
target_include_directories(torchcomms_comms_hccl PRIVATE
    ${ROOT}
    ${HCCL_INCLUDE}
    ${TORCH_NPU_INCLUDE}
    ${Python3_INCLUDE_DIRS}
)
target_compile_features(torchcomms_comms_hccl PRIVATE cxx_std_20)
target_link_libraries(torchcomms_comms_hccl PRIVATE
    glog::glog_headers
    fmt::fmt
    torchcomms
    ${TORCH_LIBRARIES}
    ${TORCH_PYTHON_LIB}
    ${HCCL_SHARED_LIB}
    ${ACL_SHARED_LIB}
    ${TORCH_NPU_LIB}
)
