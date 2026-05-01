#include <tinynccl.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <rank 0|1> <peer_host> <backend tcp|verbs> [iterations] [count]\n", argv[0]);
        return 1;
    }
    int rank = std::atoi(argv[1]);
    const char* peer = argv[2];
    const char* backend_str = argv[3];
    int iters = (argc > 4) ? std::atoi(argv[4]) : 100;
    size_t count = (argc > 5) ? static_cast<size_t>(std::atol(argv[5])) : 262144;

    tinynccl::Backend backend = tinynccl::Backend::TCP;
    if (std::strcmp(backend_str, "verbs") == 0) backend = tinynccl::Backend::Verbs;
    else if (std::strcmp(backend_str, "tcp") != 0) {
        std::fprintf(stderr, "backend must be tcp or verbs\n");
        return 1;
    }

    auto comm = tinynccl::Comm::create(rank, 2, peer, 5000, backend);
    if (!comm) return 1;

    std::vector<float> data(count);
    std::vector<float> backup(count);
    for (size_t i = 0; i < count; ++i) backup[i] = (rank + 1) * 0.001f * i;

    std::printf("rank %d: backend=%s iters=%d count=%zu (%.1f MB per buffer)\n",
                rank, backend_str, iters, count, count * sizeof(float) / 1e6);

    auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < iters; ++it) {
        std::memcpy(data.data(), backup.data(), count * sizeof(float));
        if (comm->all_reduce(data.data(), count) != 0) {
            std::fprintf(stderr, "rank %d: all_reduce failed at iter %d\n", rank, it);
            return 1;
        }
        for (size_t i = 0; i < count; ++i) {
            float expected = (1 * 0.001f * i) + (2 * 0.001f * i);
            if (std::fabs(data[i] - expected) > 1e-3f) {
                std::fprintf(stderr, "rank %d iter %d: mismatch at %zu (got %.6f, expected %.6f)\n",
                             rank, it, i, data[i], expected);
                return 1;
            }
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();
    double bytes_per_iter = count * sizeof(float);
    double mb_per_sec = (iters * bytes_per_iter / 1e6) / sec;

    std::printf("rank %d: %d iters ok in %.3fs (%.1f MB/s end-to-end)\n",
                rank, iters, sec, mb_per_sec);
    return 0;
}
