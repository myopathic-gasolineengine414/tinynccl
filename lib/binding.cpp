#include <torch/extension.h>
#include "tinynccl.h"

#include <memory>
#include <stdexcept>
#include <string>

namespace py = pybind11;

class PyComm {
public:
    PyComm(int rank, int world_size,
           const std::string& peer_host, int port,
           const std::string& backend) {
        tinynccl::Backend b = tinynccl::Backend::TCP;
        if (backend == "verbs") b = tinynccl::Backend::Verbs;
        else if (backend != "tcp") throw std::invalid_argument("backend must be tcp or verbs");

        comm_ = tinynccl::Comm::create(rank, world_size, peer_host, port, b);
        if (!comm_) throw std::runtime_error("tinynccl::Comm::create failed");
    }

    void all_reduce(torch::Tensor tensor) {
        TORCH_CHECK(tensor.is_contiguous(), "tensor must be contiguous");
        TORCH_CHECK(tensor.is_cpu(), "tensor must be on CPU (no GPU support yet)");
        TORCH_CHECK(tensor.dtype() == torch::kFloat32, "tensor must be float32");

        if (comm_->all_reduce(tensor.data_ptr(),
                              static_cast<size_t>(tensor.numel())) != 0) {
            throw std::runtime_error("all_reduce failed");
        }
    }

    int rank() const { return comm_->rank(); }
    int world_size() const { return comm_->world_size(); }

private:
    std::unique_ptr<tinynccl::Comm> comm_;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "tinynccl PyTorch bindings (Phase 3.1)";

    py::class_<PyComm>(m, "Comm")
        .def(py::init<int, int, std::string, int, std::string>(),
             py::arg("rank"),
             py::arg("world_size"),
             py::arg("peer_host"),
             py::arg("port"),
             py::arg("backend") = "tcp")
        .def("all_reduce", &PyComm::all_reduce, py::arg("tensor"))
        .def_property_readonly("rank", &PyComm::rank)
        .def_property_readonly("world_size", &PyComm::world_size);
}
