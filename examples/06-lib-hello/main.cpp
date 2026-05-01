#include <tinynccl.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <rank 0|1> <peer_host>\n", argv[0]);
        return 1;
    }
    int rank = std::atoi(argv[1]);
    const char* peer = argv[2];

    auto comm = tinynccl::Comm::create(rank, 2, peer, 5000);
    if (!comm) return 1;

    char buf[64] = {0};
    if (rank == 0) {
        comm->recv(buf, sizeof(buf));
        std::printf("rank 0: got \"%s\"\n", buf);
    } else {
        std::strncpy(buf, "hello from rank 1", sizeof(buf) - 1);
        comm->send(buf, sizeof(buf));
        std::printf("rank 1: sent \"%s\"\n", buf);
    }
    return 0;
}
