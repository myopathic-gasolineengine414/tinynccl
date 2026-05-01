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
    TinyncclWork(std::vector<at::Tensor> result = {}) {
        future_ = c10::make_intrusive<c10::ivalue::Future>(
            c10::ListType::create(c10::TensorType::get()));
        future_->markCompleted(c10::IValue(result));
    }
    bool isCompleted() override { return true; }
    bool isSuccess() const override { return true; }
    bool wait(std::chrono::milliseconds /*unused*/ = std::chrono::milliseconds(0)) override {
        return true;
    }
    c10::intrusive_ptr<c10::ivalue::Future> getFuture() override {
        return future_;
    }

private:
    c10::intrusive_ptr<c10::ivalue::Future> future_;
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
        return c10::make_intrusive<TinyncclWork>(tensors);
    }

    // DDP uses allgather to verify parameter shapes across ranks at init time.
    // 2-rank only: each rank exchanges its bytes with the peer; outputs[0][i]
    // ends up with rank i's data. Works on any dtype since we only move bytes.
    c10::intrusive_ptr<c10d::Work> allgather(
        std::vector<std::vector<at::Tensor>>& outputs,
        std::vector<at::Tensor>& inputs,
        const c10d::AllgatherOptions& /*opts*/ = c10d::AllgatherOptions()) override {
        TORCH_CHECK(inputs.size() == 1, "tinynccl: allgather one input tensor");
        TORCH_CHECK(outputs.size() == 1, "tinynccl: allgather one output list");
        TORCH_CHECK(outputs[0].size() == 2, "tinynccl: allgather requires world_size=2");

        auto& input = inputs[0];
        TORCH_CHECK(input.is_contiguous() && input.is_cpu(),
                    "tinynccl: allgather input must be contiguous CPU");

        const int rank = comm_->rank();
        const int peer = 1 - rank;
        const size_t bytes = static_cast<size_t>(input.numel()) * input.element_size();

        outputs[0][rank].copy_(input);

        if (rank == 0) {
            if (comm_->send(input.data_ptr(), bytes) != 0) throw std::runtime_error("send");
            if (comm_->recv(outputs[0][peer].data_ptr(), bytes) != 0) throw std::runtime_error("recv");
        } else {
            if (comm_->recv(outputs[0][peer].data_ptr(), bytes) != 0) throw std::runtime_error("recv");
            if (comm_->send(input.data_ptr(), bytes) != 0) throw std::runtime_error("send");
        }
        return c10::make_intrusive<TinyncclWork>(std::vector<at::Tensor>{});
    }

    c10::intrusive_ptr<c10d::Work> _allgather_base(
        at::Tensor& output,
        at::Tensor& input,
        const c10d::AllgatherOptions& /*opts*/ = c10d::AllgatherOptions()) override {
        TORCH_CHECK(input.is_contiguous() && output.is_contiguous(),
                    "tinynccl: contiguous required");
        TORCH_CHECK(input.is_cpu() && output.is_cpu(),
                    "tinynccl: CPU only");

        const int rank = comm_->rank();
        const size_t in_bytes = static_cast<size_t>(input.numel()) * input.element_size();

        // Place my chunk into output at offset rank * in_bytes.
        std::memcpy(static_cast<char*>(output.data_ptr()) + rank * in_bytes,
                    input.data_ptr(), in_bytes);

        const int peer = 1 - rank;
        char* peer_slot = static_cast<char*>(output.data_ptr()) + peer * in_bytes;

        if (rank == 0) {
            if (comm_->send(input.data_ptr(), in_bytes) != 0) throw std::runtime_error("send");
            if (comm_->recv(peer_slot, in_bytes) != 0) throw std::runtime_error("recv");
        } else {
            if (comm_->recv(peer_slot, in_bytes) != 0) throw std::runtime_error("recv");
            if (comm_->send(input.data_ptr(), in_bytes) != 0) throw std::runtime_error("send");
        }
        return c10::make_intrusive<TinyncclWork>(std::vector<at::Tensor>{});
    }

    c10::intrusive_ptr<c10d::Work> broadcast(
        std::vector<at::Tensor>& tensors,
        const c10d::BroadcastOptions& opts = c10d::BroadcastOptions()) override {
        TORCH_CHECK(tensors.size() == 1, "tinynccl: broadcast one tensor");
        auto& t = tensors[0];
        TORCH_CHECK(t.is_contiguous() && t.is_cpu(),
                    "tinynccl: contiguous CPU tensor required");
        const size_t bytes = static_cast<size_t>(t.numel()) * t.element_size();
        const int rank = comm_->rank();

        if (rank == opts.rootRank) {
            if (comm_->send(t.data_ptr(), bytes) != 0) throw std::runtime_error("send");
        } else {
            if (comm_->recv(t.data_ptr(), bytes) != 0) throw std::runtime_error("recv");
        }
        return c10::make_intrusive<TinyncclWork>(std::vector<at::Tensor>{});
    }

    c10::intrusive_ptr<c10d::Work> barrier(
        const c10d::BarrierOptions& /*opts*/ = c10d::BarrierOptions()) override {
        // Cheap barrier: exchange one byte both ways.
        char b = 0;
        const int rank = comm_->rank();
        if (rank == 0) {
            if (comm_->send(&b, 1) != 0) throw std::runtime_error("send");
            if (comm_->recv(&b, 1) != 0) throw std::runtime_error("recv");
        } else {
            if (comm_->recv(&b, 1) != 0) throw std::runtime_error("recv");
            if (comm_->send(&b, 1) != 0) throw std::runtime_error("send");
        }
        return c10::make_intrusive<TinyncclWork>(std::vector<at::Tensor>{});
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
