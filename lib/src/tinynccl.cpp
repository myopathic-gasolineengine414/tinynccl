#include "tinynccl.h"
#include "transport.h"

#include <cstdio>

namespace tinynccl {

namespace {

class CommImpl : public Comm {
public:
    CommImpl(int rank, int world_size, std::unique_ptr<Transport> t)
        : transport_(std::move(t)) {
        rank_ = rank;
        world_size_ = world_size;
    }

    int send(const void* buf, size_t bytes) override {
        return transport_->send(buf, bytes);
    }

    int recv(void* buf, size_t bytes) override {
        return transport_->recv(buf, bytes);
    }

private:
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
            std::fprintf(stderr, "tinynccl: verbs backend not yet wired up\n");
            return nullptr;
    }

    if (t->establish(rank, peer_host, port) != 0) {
        return nullptr;
    }

    return std::make_unique<CommImpl>(rank, world_size, std::move(t));
}

}
