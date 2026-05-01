# 0.4: cuda + verbs

Combined CUDA + verbs example. The client kernel writes a known pattern to a `cudaMalloc`'d float buffer; the server eventually has the same data in its own GPU buffer. End-to-end GPU → wire → GPU.

The first thing each side tries is registering the `cudaMalloc`'d device pointer directly with `ibv_reg_mr`. This is the strict GPUDirect RDMA path, and on this hardware it fails. softRoCE has no DMA hardware (the kernel constructs packets in software), and the GTX 1080 Ti is a consumer GeForce that doesn't expose its memory to PCIe peers.

The fallback is what production HPC code does when GPUDirect isn't available: register pinned host memory (`cudaMallocHost`) and stage the GPU↔host transfer explicitly with `cudaMemcpy`. The path becomes:

    GPU device buf → pinned host buf → verbs SEND → wire → verbs RECV → pinned host buf → GPU device buf

On real hardware (Mellanox + datacenter GPU + `nvidia_peermem`) the strict path works and the bounces go away.

Run:

    make run
