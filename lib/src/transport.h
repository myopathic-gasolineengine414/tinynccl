#pragma once

#include <cstddef>
#include <memory>
#include <string>

namespace tinynccl {

class Transport {
public:
    virtual ~Transport() = default;
    virtual int establish(int rank, const std::string& peer_host, int port) = 0;
    virtual int send(const void* buf, size_t bytes) = 0;
    virtual int recv(void* buf, size_t bytes) = 0;
};

std::unique_ptr<Transport> make_tcp_transport();

}
