#include <tinynccl.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <rank 0|1> <peer_host>\n", argv[0]);
        return 1;
    }
    int rank = std::atoi(argv[1]);
    const char* peer = argv[2];

    auto comm = tinynccl::Comm::create(rank, 2, peer, 5000);
    if (!comm) return 1;

    const size_t n = 8;
    std::vector<float> data(n);
    for (size_t i = 0; i < n; ++i) {
        data[i] = (rank + 1) * 100.0f + static_cast<float>(i);
    }

    std::printf("rank %d before: ", rank);
    for (float v : data) std::printf("%.1f ", v);
    std::printf("\n");

    if (comm->all_reduce(data.data(), n) != 0) {
        std::fprintf(stderr, "all_reduce failed\n");
        return 1;
    }

    std::printf("rank %d after:  ", rank);
    for (float v : data) std::printf("%.1f ", v);
    std::printf("\n");

    for (size_t i = 0; i < n; ++i) {
        float expected = (100.0f + i) + (200.0f + i);
        if (std::fabs(data[i] - expected) > 1e-3f) {
            std::fprintf(stderr, "rank %d: mismatch at %zu (got %.1f, expected %.1f)\n",
                         rank, i, data[i], expected);
            return 1;
        }
    }
    std::printf("rank %d: ok\n", rank);
    return 0;
}
