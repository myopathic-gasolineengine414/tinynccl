#include "transport.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>

namespace tinynccl {

namespace {

class TcpTransport : public Transport {
public:
    ~TcpTransport() override {
        if (fd_ >= 0) ::close(fd_);
    }

    int establish(int rank, const std::string& peer_host, int port) override {
        if (rank == 0) {
            int listen_fd = ::socket(AF_INET, SOCK_STREAM, 0);
            if (listen_fd < 0) { ::perror("socket"); return -1; }
            int opt = 1;
            ::setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
            sockaddr_in addr{};
            addr.sin_family = AF_INET;
            addr.sin_port = htons(port);
            addr.sin_addr.s_addr = INADDR_ANY;
            if (::bind(listen_fd, (sockaddr*)&addr, sizeof(addr)) < 0) {
                ::perror("bind"); ::close(listen_fd); return -1;
            }
            if (::listen(listen_fd, 1) < 0) {
                ::perror("listen"); ::close(listen_fd); return -1;
            }
            fd_ = ::accept(listen_fd, nullptr, nullptr);
            ::close(listen_fd);
            if (fd_ < 0) { ::perror("accept"); return -1; }
        } else {
            fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
            if (fd_ < 0) { ::perror("socket"); return -1; }
            sockaddr_in addr{};
            addr.sin_family = AF_INET;
            addr.sin_port = htons(port);
            if (::inet_pton(AF_INET, peer_host.c_str(), &addr.sin_addr) <= 0) {
                std::fprintf(stderr, "invalid peer host: %s\n", peer_host.c_str());
                return -1;
            }
            if (::connect(fd_, (sockaddr*)&addr, sizeof(addr)) < 0) {
                ::perror("connect"); return -1;
            }
        }
        return 0;
    }

    int send(const void* buf, size_t bytes) override {
        size_t sent = 0;
        const char* p = static_cast<const char*>(buf);
        while (sent < bytes) {
            ssize_t n = ::send(fd_, p + sent, bytes - sent, 0);
            if (n <= 0) { ::perror("send"); return -1; }
            sent += static_cast<size_t>(n);
        }
        return 0;
    }

    int recv(void* buf, size_t bytes) override {
        size_t got = 0;
        char* p = static_cast<char*>(buf);
        while (got < bytes) {
            ssize_t n = ::recv(fd_, p + got, bytes - got, 0);
            if (n <= 0) { ::perror("recv"); return -1; }
            got += static_cast<size_t>(n);
        }
        return 0;
    }

private:
    int fd_ = -1;
};

}

std::unique_ptr<Transport> make_tcp_transport() {
    return std::make_unique<TcpTransport>();
}

}
