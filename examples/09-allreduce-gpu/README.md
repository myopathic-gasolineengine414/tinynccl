# 2.3: allreduce-gpu

Two processes, both `cudaMalloc` on the same GTX 1080 Ti. Each rank initializes its GPU buffer via a kernel, runs `comm->all_reduce(...)` over the verbs backend with pinned host as staging, copies the result back to GPU memory, validates.

The data path:

    GPU (rank N) -> pinned host -> verbs -> wire -> verbs -> pinned host -> GPU (rank N)

Same picture as `04-cuda-verbs` but now the operation in the middle is a real collective (all-reduce) instead of a single SEND/RECV. On hardware with GPUDirect RDMA the pinned-host bounces would disappear and verbs would DMA straight from GPU memory.

Two CUDA contexts on one physical GPU works because each process gets its own context with isolated memory; the 1080 Ti has 11 GB of VRAM and these buffers are tiny.

Run:

    make run
