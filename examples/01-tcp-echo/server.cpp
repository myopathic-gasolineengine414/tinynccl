#include <iostream>
#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>

int main(int argc, char** argv) {
    int port = (argc > 1) ? std::atoi(argv[1]) : 5000;

    int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd < 0) { perror("socket"); return 1; }

    int opt = 1;
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = INADDR_ANY;

    if (bind(listen_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        perror("bind"); return 1;
    }
    if (listen(listen_fd, 1) < 0) { perror("listen"); return 1; }

    std::cout << "server: listening on port " << port << "\n";

    int conn_fd = accept(listen_fd, nullptr, nullptr);
    if (conn_fd < 0) { perror("accept"); return 1; }

    char buf[1024];
    ssize_t n = recv(conn_fd, buf, sizeof(buf) - 1, 0);
    if (n < 0) { perror("recv"); return 1; }
    buf[n] = 0;

    std::cout << "server: received " << n << " bytes: \"" << buf << "\"\n";

    if (send(conn_fd, buf, n, 0) != n) {
        perror("send"); return 1;
    }
    std::cout << "server: echoed back\n";

    close(conn_fd);
    close(listen_fd);
    return 0;
}
