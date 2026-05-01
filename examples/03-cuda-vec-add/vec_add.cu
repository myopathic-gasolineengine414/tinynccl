#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CHECK_CUDA(expr) do { \
    cudaError_t err = (expr); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        std::exit(1); \
    } \
} while (0)

__global__ void vec_add(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    const int n = 1 << 20;
    const size_t bytes = n * sizeof(float);

    float *h_a = (float*)std::malloc(bytes);
    float *h_b = (float*)std::malloc(bytes);
    float *h_c = (float*)std::malloc(bytes);
    for (int i = 0; i < n; i++) {
        h_a[i] = i * 0.5f;
        h_b[i] = i * 2.0f;
    }

    float *d_a, *d_b, *d_c;
    CHECK_CUDA(cudaMalloc(&d_a, bytes));
    CHECK_CUDA(cudaMalloc(&d_b, bytes));
    CHECK_CUDA(cudaMalloc(&d_c, bytes));

    CHECK_CUDA(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    vec_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    int errors = 0;
    for (int i = 0; i < n; i++) {
        float expected = h_a[i] + h_b[i];
        if (std::fabs(h_c[i] - expected) > 1e-3f) errors++;
    }

    if (errors == 0) {
        printf("vec_add: %d elements verified, c[0]=%.2f, c[%d]=%.2f\n",
               n, h_c[0], n - 1, h_c[n - 1]);
    } else {
        printf("vec_add: %d errors out of %d\n", errors, n);
        return 1;
    }

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    std::free(h_a); std::free(h_b); std::free(h_c);
    return 0;
}
