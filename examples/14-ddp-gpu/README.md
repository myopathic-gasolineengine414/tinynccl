# 3.3: ddp on gpu

Real GPU `DistributedDataParallel` training with tinynccl as the backend. Two PyTorch processes both target the same GTX 1080 Ti via separate CUDA contexts. The model lives on `cuda:0`, the data lives on `cuda:0`, gradients are computed on GPU, and the gradient sync flows through tinynccl over softRoCE.

The data path for each `loss.backward()` step:

    CUDA grad tensor → t.cpu() [cudaMemcpy GPU→host]
                     → tinynccl::all_reduce [over softRoCE]
                     → t.copy_(host) [cudaMemcpy host→GPU]

The host bounce is required because softRoCE is a kernel implementation that can't register cudaMalloc'd memory, and the 1080 Ti is a consumer GeForce that doesn't expose its memory to PCIe peers (no GPUDirect RDMA). On real datacenter hardware (Mellanox ConnectX + Tesla/Quadro + nvidia_peermem) the host bounce disappears and verbs DMAs straight from GPU memory.

## Run

```
./run.sh        # verbs (default)
./run.sh tcp    # tcp transport
```

## What this proves

- The c10d::Backend handles CUDA tensors correctly (staging in `allreduce`, `allgather`, `_allgather_base`, `broadcast`).
- DDP can route gradient sync through tinynccl on real GPU tensors.
- After 5 SGD steps with different data per rank, both ranks have identical weights at machine precision (max diff `0.00e+00`).
- The headline claim — "trained a model with my custom RDMA-based backend on real GPUs" — is now factually true.
