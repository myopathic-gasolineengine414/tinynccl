# Milestone 0.4: CUDA buffer registered with verbs

GPU-aware verbs SEND/RECV. The client initializes a `cudaMalloc`'d float buffer via a CUDA kernel, the server eventually receives that data into its own `cudaMalloc`'d device buffer. Demonstrates the full GPU → network → GPU data path on softRoCE.

## What this demonstrates about GPUDirect RDMA on this hardware

The first thing each side tries is **strict-mode GPU memory registration**: passing a `cudaMalloc`'d device pointer directly to `ibv_reg_mr`. On a real Mellanox ConnectX + Tesla/Quadro stack, that succeeds and the NIC DMAs straight from GPU memory.

On this box it fails. softRoCE has no DMA hardware (the kernel constructs packets in software), and the GTX 1080 Ti is a consumer GeForce card that historically does not expose its memory to PCIe peers. We expect the registration to fail and document the actual error.

The fallback used here is the same one production HPC code uses when GPUDirect isn't available: **register pinned host memory** (`cudaMallocHost`) and stage the GPU↔host copy explicitly via `cudaMemcpy`. Pinned host memory:
- is allocated by CUDA but lives in regular host RAM
- is page-locked, so the kernel can DMA from it without unpinning
- can be registered with `ibv_reg_mr` like any other host buffer
- is the source/sink for very fast `cudaMemcpy` to/from the GPU

The data path becomes: `GPU device buffer → pinned host buffer → verbs SEND → wire → verbs RECV → pinned host buffer → GPU device buffer`.

## Build and run

```
make run
```

## What this proves

- The CUDA toolchain and verbs userland coexist in a single binary linked with both `libcudart` and `libibverbs`.
- We can detect at runtime whether GPU memory registration is supported and gracefully fall back when it isn't.
- The full GPU-to-GPU pipeline works end-to-end: a kernel writes the data, verbs carries it across, the receiver lands it back in GPU memory.
- The expected output values (1.5, 3.0, ..., 24.0) survive all four memory copies plus the verbs round trip.

## What changes on real hardware

On a system with Mellanox ConnectX + Tesla/Quadro/datacenter GPU + `nvidia_peermem`:
- The strict-mode `ibv_reg_mr` on the device pointer succeeds.
- Both `cudaMallocHost` allocations and both `cudaMemcpy` calls go away.
- The NIC DMAs directly from GPU memory across PCIe peer-to-peer.
- Latency drops dramatically; the bounce is the dominant cost on small messages.

The library code we'll build later abstracts this so the same call site uses GPUDirect when available and falls back to pinned-host staging when not.
