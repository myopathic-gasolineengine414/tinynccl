#include <iostream>
#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

int main(int argc, char** argv) {
    const char* host = (argc > 1) ? argv[1] : "127.0.0.1";
    int port = (argc > 2) ? std::atoi(argv[2]) : 5000;
    const char* msg = (argc > 3) ? argv[3] : "hello, tinynccl";

    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) { perror("socket"); return 1; }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    if (inet_pton(AF_INET, host, &addr.sin_addr) <= 0) {
        std::cerr << "client: invalid address: " << host << "\n";
        return 1;
    }

    if (connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        perror("connect"); return 1;
    }

    size_t len = std::strlen(msg);
    std::cout << "client: connected to " << host << ":" << port
              << ", sending \"" << msg << "\"\n";

    if (send(fd, msg, len, 0) != static_cast<ssize_t>(len)) {
        perror("send"); return 1;
    }

    char buf[1024];
    ssize_t n = recv(fd, buf, sizeof(buf) - 1, 0);
    if (n < 0) { perror("recv"); return 1; }
    buf[n] = 0;

    std::cout << "client: received " << n << " bytes: \"" << buf << "\"\n";

    close(fd);
    return 0;
}
