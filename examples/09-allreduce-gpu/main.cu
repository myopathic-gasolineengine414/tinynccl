#include <tinynccl.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK_CUDA(expr) do { \
    cudaError_t e = (expr); \
    if (e != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                     cudaGetErrorString(e)); \
        std::exit(1); \
    } \
} while (0)

__global__ void init_kernel(float* buf, int n, int rank) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = (rank + 1) * 100.0f + static_cast<float>(i);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <rank 0|1> <peer_host>\n", argv[0]);
        return 1;
    }
    int rank = std::atoi(argv[1]);
    const char* peer = argv[2];

    auto comm = tinynccl::Comm::create(rank, 2, peer, 5000, tinynccl::Backend::Verbs);
    if (!comm) return 1;

    const size_t n = 8;
    const size_t bytes = n * sizeof(float);

    float* d_buf;
    CHECK_CUDA(cudaMalloc(&d_buf, bytes));

    init_kernel<<<1, n>>>(d_buf, n, rank);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    float* h_buf;
    CHECK_CUDA(cudaMallocHost(&h_buf, bytes));
    CHECK_CUDA(cudaMemcpy(h_buf, d_buf, bytes, cudaMemcpyDeviceToHost));

    std::printf("rank %d before: ", rank);
    for (size_t i = 0; i < n; i++) std::printf("%.1f ", h_buf[i]);
    std::printf("\n");

    if (comm->all_reduce(h_buf, n) != 0) {
        std::fprintf(stderr, "all_reduce failed\n");
        return 1;
    }

    CHECK_CUDA(cudaMemcpy(d_buf, h_buf, bytes, cudaMemcpyHostToDevice));

    std::vector<float> verify(n);
    CHECK_CUDA(cudaMemcpy(verify.data(), d_buf, bytes, cudaMemcpyDeviceToHost));

    std::printf("rank %d after:  ", rank);
    for (float v : verify) std::printf("%.1f ", v);
    std::printf("\n");

    for (size_t i = 0; i < n; ++i) {
        float expected = (100.0f + i) + (200.0f + i);
        if (std::fabs(verify[i] - expected) > 1e-3f) {
            std::fprintf(stderr, "rank %d: mismatch at %zu (got %.1f, expected %.1f)\n",
                         rank, i, verify[i], expected);
            return 1;
        }
    }
    std::printf("rank %d: ok (gpu -> verbs -> gpu)\n", rank);

    CHECK_CUDA(cudaFreeHost(h_buf));
    CHECK_CUDA(cudaFree(d_buf));
    return 0;
}
