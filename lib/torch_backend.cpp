#include <torch/extension.h>
#include <pybind11/chrono.h>
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>

#include "tinynccl.h"

#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>

namespace py = pybind11;

class TinyncclWork : public c10d::Work {
public:
    TinyncclWork() = default;
    bool isCompleted() override { return true; }
    bool isSuccess() const override { return true; }
    bool wait(std::chrono::milliseconds /*unused*/ = std::chrono::milliseconds(0)) override {
        return true;
    }
};

class TinyncclBackend : public c10d::Backend {
public:
    TinyncclBackend(int rank, int size,
                    const std::string& peer_host, int port,
                    const std::string& backend_name)
        : c10d::Backend(rank, size) {
        tinynccl::Backend b = (backend_name == "verbs")
            ? tinynccl::Backend::Verbs
            : tinynccl::Backend::TCP;
        comm_ = tinynccl::Comm::create(rank, size, peer_host, port, b);
        if (!comm_) {
            throw std::runtime_error("tinynccl::Comm::create failed");
        }
    }

    const std::string getBackendName() const override {
        return "tinynccl";
    }

    c10::intrusive_ptr<c10d::Work> allreduce(
        std::vector<at::Tensor>& tensors,
        const c10d::AllreduceOptions& opts = c10d::AllreduceOptions()) override {
        TORCH_CHECK(tensors.size() == 1,
                    "tinynccl: allreduce supports a single tensor only");
        auto& t = tensors[0];
        TORCH_CHECK(t.is_contiguous(), "tinynccl: tensor must be contiguous");
        TORCH_CHECK(t.is_cpu(), "tinynccl: tensor must be on CPU");
        TORCH_CHECK(t.dtype() == torch::kFloat32, "tinynccl: tensor must be float32");
        TORCH_CHECK(opts.reduceOp == c10d::ReduceOp::SUM, "tinynccl: only SUM is supported");

        if (comm_->all_reduce(t.data_ptr(), static_cast<size_t>(t.numel())) != 0) {
            throw std::runtime_error("tinynccl::Comm::all_reduce failed");
        }
        return c10::make_intrusive<TinyncclWork>();
    }

private:
    std::unique_ptr<tinynccl::Comm> comm_;
};

// Factory invoked by torch.distributed.Backend.register_backend.
// PyTorch passes (prefix_store, rank, size, timeout); we read connection
// info from environment variables for now (MASTER_ADDR / MASTER_PORT /
// optional TINYNCCL_BACKEND).
c10::intrusive_ptr<c10d::Backend> create_tinynccl_backend(
    const c10::intrusive_ptr<c10d::Store>& /*store*/,
    int rank,
    int size,
    const std::chrono::duration<float>& /*timeout*/) {

    const char* master_addr_env = std::getenv("MASTER_ADDR");
    const char* master_port_env = std::getenv("MASTER_PORT");
    const char* backend_env = std::getenv("TINYNCCL_BACKEND");

    std::string addr = master_addr_env ? master_addr_env : "127.0.0.1";
    int port = master_port_env ? std::atoi(master_port_env) : 5000;
    std::string backend_name = backend_env ? backend_env : "tcp";

    return c10::make_intrusive<TinyncclBackend>(
        rank, size, addr, port, backend_name);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("create_backend", &create_tinynccl_backend);
}
