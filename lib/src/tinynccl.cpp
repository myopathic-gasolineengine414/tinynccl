#include "tinynccl.h"
#include "transport.h"

#include <cstdio>
#include <vector>

namespace tinynccl {

namespace {

class CommImpl : public Comm {
public:
    CommImpl(int rank, int world_size, std::unique_ptr<Transport> t) {
        rank_ = rank;
        world_size_ = world_size;
        transport_ = std::move(t);
    }

    int send(const void* buf, size_t bytes) override {
        return transport_->send(buf, bytes);
    }

    int recv(void* buf, size_t bytes) override {
        return transport_->recv(buf, bytes);
    }

    int all_reduce(void* buf, size_t count, DataType dtype, ReduceOp op) override {
        if (dtype != DataType::Float32 || op != ReduceOp::Sum) {
            std::fprintf(stderr, "tinynccl: only Float32/Sum supported\n");
            return -1;
        }
        if (world_size_ != 2) {
            std::fprintf(stderr, "tinynccl: all_reduce only supports world_size=2\n");
            return -1;
        }

        const size_t bytes = count * sizeof(float);
        if (peer_buf_.size() < count) peer_buf_.resize(count);

        // Alternate to avoid deadlock on synchronous transports.
        if (rank_ == 0) {
            if (transport_->send(buf, bytes)) return -1;
            if (transport_->recv(peer_buf_.data(), bytes)) return -1;
        } else {
            if (transport_->recv(peer_buf_.data(), bytes)) return -1;
            if (transport_->send(buf, bytes)) return -1;
        }

        float* dst = static_cast<float*>(buf);
        for (size_t i = 0; i < count; ++i) {
            dst[i] += peer_buf_[i];
        }
        return 0;
    }

private:
    // Order matters: peer_buf_ must outlive transport_ so the cached MR for
    // peer_buf_.data() can be deregistered before its backing memory frees.
    std::vector<float> peer_buf_;
    std::unique_ptr<Transport> transport_;
};

}

std::unique_ptr<Comm> Comm::create(
    int rank, int world_size,
    const std::string& peer_host, int port,
    Backend backend) {

    if (world_size != 2) {
        std::fprintf(stderr, "tinynccl: only world_size=2 supported for now\n");
        return nullptr;
    }

    std::unique_ptr<Transport> t;
    switch (backend) {
        case Backend::TCP:
            t = make_tcp_transport();
            break;
        case Backend::Verbs:
            t = make_verbs_transport();
            break;
    }

    if (t->establish(rank, peer_host, port) != 0) {
        return nullptr;
    }

    return std::make_unique<CommImpl>(rank, world_size, std::move(t));
}

}
