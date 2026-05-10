// Copyright (c) 2026 Shan Shan.
//
// Static-init register_backend lives in TorchCommHCCL.cpp; this module
// declares the Python type with TorchCommBackend as parent so pybind11
// preserves subclass identity through cast / isinstance — aligning with
// nccl / gloo / rccl / rcclx / ncclx / xccl pybind pattern (PR #2080).

#include <pybind11/chrono.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/utils/pybind.h>

#include "comms/torchcomms/hccl/TorchCommHCCL.hpp"

namespace py = pybind11;
using namespace torch::comms;

PYBIND11_MODULE(_comms_hccl, m, py::mod_gil_not_used()) {
  m.doc() = "HCCL specific python bindings for TorchComm";

  py::class_<TorchCommHCCL, TorchCommBackend, std::shared_ptr<TorchCommHCCL>>(
      m, "TorchCommHCCL");
}
