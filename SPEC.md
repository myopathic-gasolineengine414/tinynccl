# tinynccl — A Tiny NCCL from Scratch

A from-scratch C++/CUDA implementation of GPU collective communication operations (specifically all-reduce), built on libibverbs over softRoCE, integrated with PyTorch as a custom `ProcessGroup` backend. Trains a real model end-to-end across two physical machines using transport code I wrote myself.

---

## Why this exists

This project exists to build deep, demonstrable competence in GPU networking and distributed AI systems for a NVIDIA Networking Systems Software Engineering internship application. The role wants engineers who can write C/C++ for ConnectX SmartNICs and BlueField DPUs, understand RDMA / RoCE, debug end-to-end perf in distributed training, and reason about GPU-to-NIC data paths. This project produces hands-on experience with every one of those bullets and a public artifact (the GitHub repo + writeup) that proves it.

Owner: Jack Lutz (jwlutz / jackwlutz1@gmail.com). UCLA Stats & Data Science, class of 2027.

## Goal

By project completion, a single command on the Mac launches distributed training across the two project boxes. Gradient all-reduce flows through the C++ library in this repo over libibverbs over softRoCE over a direct ethernet cable. A small transformer (~50M params) trains on TinyShakespeare across both GPUs to convergence. A benchmark script compares throughput and latency vs real NCCL on the same hardware/workload, with an honest writeup of the gap and its causes.

The README of the repo at the end should let a NVIDIA recruiter understand in 60 seconds that the candidate has actually walked the full GPU-to-wire data path, not just used a high-level library.

## Hardware

| Box           | Role         | GPU       | OS              | WiFi IP         | Direct-cable IP |
|---------------|--------------|-----------|-----------------|-----------------|-----------------|
| Mac           | dev workstation | n/a    | macOS           | (varies)        | n/a             |
| `personal`    | project node 0 | GTX 1080 Ti (Pascal) | Ubuntu (already installed) | 192.168.10.206  | 10.0.0.2        |
| (laptop, TBD) | project node 1 | RTX 4070 (Ada) | Ubuntu (TODO: dual-boot install) | TBD          | 10.0.0.1        |

- Mac is the dev workstation only. ssh + VS Code Remote-SSH into the project boxes. Mac never participates in project traffic.
- The two project boxes are connected with a direct ethernet cable, no switch between them. A separate WiFi connection on each box gives the Mac access.
- Project traffic flows ONLY over the direct cable (10.0.0.0/24 subnet). This is critical for clean perf measurements.
- Static IPs on the cable interfaces must be configured manually on both boxes. WiFi IPs come from DHCP.

## Software stack

- Ubuntu 24.04 LTS on both project boxes (currently only on `personal`)
- Linux kernel with `rdma_rxe` (softRoCE) module support — included in stock Ubuntu kernels
- userland: `libibverbs-dev`, `librdmacm-dev`, `rdma-core`
- CUDA Toolkit 12.x (compatible with both Pascal and Ada)
- C++17 via `g++` or `clang++`
- CMake or plain Makefile
- PyTorch 2.x (Phase 3+)
- Python 3.11+ for test harness

WSL2 was considered for the laptop and rejected: virtualized network stack pollutes RDMA semantics and perf measurements. Native Linux on both boxes is required.

## Architecture

### Library API (target)

```cpp
namespace tinynccl {
    void init(int rank, int world_size, const std::string& peer_addr);
    void all_reduce(float* gpu_buffer, size_t count, ReduceOp op);
    void destroy();
}
```

### Transport backends (pluggable)

- `tcp`: vanilla BSD sockets. Baseline for correctness and a sanity-check perf comparison. Pure CPU, host-staged.
- `verbs`: libibverbs over softRoCE. The production path. Registers GPU memory with `ibv_reg_mr` so the NIC can read/write CUDA buffers directly. Uses `ibv_post_send` / `ibv_post_recv` with completion queue polling.

The library picks the backend at init time. Every collective op has the same interface regardless of backend.

### Algorithm

- Phase 2a: naive 2-node all-reduce. Each rank sends its full buffer to the peer, sums received data into local buffer, done. O(N) data transferred per node.
- Phase 2b: ring all-reduce. Split buffer into chunks, two passes (reduce-scatter + all-gather), each node only sends/receives 2(N-1)/N of its data. Bandwidth-optimal. With 2 nodes this degenerates to roughly the naive case but the algorithmic infrastructure is in place for >2 nodes.

### PyTorch integration

A custom `ProcessGroup` backend (~200 lines C++ + Python shim) plugs `tinynccl` into `torch.distributed`. Standard DDP code paths then use it transparently:

```python
torch.distributed.init_process_group(backend="tinynccl", ...)
model = torch.nn.parallel.DistributedDataParallel(model)
```

Reference: PyTorch's `gloo` backend implementation (~3K LOC, in pytorch/torch/distributed) is the canonical model to crib structure from.

## Milestones

### Phase 0: Foundation (single-machine, ongoing while waiting for Ubuntu install on laptop)

- [x] ssh key auth from Mac to `personal`
- [x] git repo skeleton on `personal` at `~/tinynccl`
- [ ] VS Code Remote-SSH from Mac to `personal`
- [x] GitHub remote configured, first commit pushed (https://github.com/jwlutz/tinynccl, private)
- [x] `apt update && apt upgrade` on `personal` to clear pending updates
- [x] Install build dependencies: `build-essential cmake git libibverbs-dev librdmacm-dev rdma-core ibverbs-utils perftest`
- [ ] CUDA Toolkit 12.x installed and verified (deferred — NVIDIA driver also not present; install when Milestone 0.3 needs it)
- [x] Load `rdma_rxe` module (persisted via `/etc/modules-load.d/rdma_rxe.conf`)
- [x] Configure RXE device on `wlp10s0` (runtime only — needs re-add after reboot)
- [x] Verify with `ibv_devices` and `ibv_devinfo` (rxe0 ACTIVE, GUID 522e91fffe08b558)
- [x] **Milestone 0.1**: TCP echo program in C++, two processes on same machine. See `examples/01-tcp-echo/`.
- [x] **Milestone 0.2**: libibverbs hello-world in C, two processes on same machine talking via RXE loopback. See `examples/02-verbs-hello/`. SEND/RECV (not RDMA_WRITE) for the simplest semantics; covers device open, PD, MR, CQ, QP, INIT→RTR→RTS transitions, post send/recv, CQ poll.
- [x] **Milestone 0.3**: CUDA hello-world. Vector add kernel on the 1080 Ti. See `examples/03-cuda-vec-add/`. 1M elements summed and verified.
- [x] **Milestone 0.4**: GPU-aware verbs SEND/RECV. See `examples/04-cuda-verbs/`. Strict-mode `ibv_reg_mr` on a `cudaMalloc`'d pointer fails with EFAULT on softRoCE/consumer GPU as expected; fallback registers pinned host memory (`cudaMallocHost`) and stages GPU↔host via `cudaMemcpy`. Full GPU→wire→GPU data path validated end-to-end with numerical correctness.
- [x] **Milestone 0.5**: RDMA `WRITE_WITH_IMM`. See `examples/05-rdma-write/`. One-sided write — server publishes its buf addr + rkey via OOB, client posts the write directly into server memory. Server CPU doesn't participate in the data transfer. This is the operation NCCL/MPI use for bulk data; SEND/RECV is for control.

### Phase 1: Cross-machine basics (after Ubuntu lands on the laptop)

- [ ] Ubuntu install on laptop, dual-boot or external SSD
- [ ] Mirror Phase 0 setup on laptop: deps, CUDA, softRoCE
- [ ] Static IPs on direct cable: 10.0.0.1 on laptop, 10.0.0.2 on `personal`
- [ ] `ping` succeeds across cable
- [ ] ssh from Mac to laptop works
- [ ] **Milestone 1.1**: TCP echo across cable. Same code as Milestone 0.1 with different IPs.
- [ ] **Milestone 1.2**: libibverbs hello-world across cable. Same code as 0.2 with different IPs.
- [ ] **Milestone 1.3**: GPU buffer on laptop (4070), read by `personal` via verbs over cable. Validates GPU-to-network-to-GPU data path.

### Phase 2: All-reduce

- [x] **Milestone 2.1**: CPU all-reduce over TCP. Two ranks, float arrays, validate correctness. See `examples/07-allreduce-tcp/`. Naive: rank 0 sends, rank 1 sends; both sum locally. `lib/` now has `Comm::all_reduce(buf, count, dtype, op)` for Float32/Sum.
- [ ] **Milestone 2.2**: CPU all-reduce over verbs. Replace TCP transport with verbs, same correctness.
- [ ] **Milestone 2.3**: GPU all-reduce over verbs (single chunk, naive 2-node).
- [ ] **Milestone 2.4**: Ring all-reduce decomposition (chunked).
- [ ] **Milestone 2.5**: Performance instrumentation. Per-call latency, throughput, busy-wait vs blocking polling.

### Phase 3: PyTorch integration

- [ ] **Milestone 3.1**: Minimal `ProcessGroup` backend exposing only `all_reduce`.
- [ ] **Milestone 3.2**: Toy DDP training loop (e.g. MNIST MLP) using tinynccl backend.
- [ ] **Milestone 3.3**: 50M-param GPT on TinyShakespeare, two-rank training to convergence.
- [ ] **Milestone 3.4**: Numerical correctness vs NCCL on the same loop.

### Phase 4: Benchmarking and writeup

- [ ] Latency micro-benchmark (small messages) vs NCCL
- [ ] Throughput macro-benchmark (large gradients) vs NCCL
- [ ] Scaling efficiency (training step time vs single-GPU baseline)
- [ ] Profile with Nsight Systems
- [ ] README with results, architecture, "where I lose to NCCL and why"
- [ ] Optional: blog post on jwlutz.com

## Stretch goals (post-MVP)

- GPU-initiated communication via NVSHMEM-style API or kernel-driven verbs
- Tree all-reduce in addition to ring (relevant for >2 nodes if a third box becomes available)
- Gradient compression in transport layer (top-k, quantization)
- Failure injection via `tc netem`, simple recovery path
- DCQCN-style congestion control simulation

## Constraints, gotchas, and honest limitations

- **softRoCE is not real RDMA hardware.** Performance numbers will be lower than they would be on a ConnectX. The point of the project is the API, semantics, and protocol implementation, not absolute speed. The writeup must be honest about this.
- **1080 Ti is Pascal.** Supports CUDA, no Tensor Cores, no fp8. Fine for this project.
- **4070 is Ada.** Supports everything modern. Mismatched compute capability between ranks is realistic for the project's scale and not a problem.
- **WiFi is for dev access only.** All project traffic flows over the direct ethernet cable. Never benchmark over WiFi.
- **`personal` has 44 pending updates** as of project start. Run `apt upgrade` and reboot before installing softRoCE / building anything serious to avoid kernel module mismatches.
- **NCCL comparison fairness**: real NCCL will use NVLink or PCIe optimally. tinynccl will use only the verbs path. Compare apples to apples by forcing NCCL to use the same network path (set `NCCL_IB_DISABLE=0`, `NCCL_P2P_DISABLE=1` for relevant runs).

## Reference materials

- **Beej's Guide to Network Programming** — free PDF — sockets fundamentals
- **RDMA Aware Networks Programming User Manual** (Mellanox) — free PDF — verbs API canonical reference
- **NCCL technical blog posts** on developer.nvidia.com — algorithm explanations
- **Linux RDMA Subsystem documentation** (kernel.org) — softRoCE / RXE setup
- **PyTorch `gloo` backend source** (pytorch/torch/distributed) — ProcessGroup implementation reference
- **"Bringing HPC Techniques to Deep Learning"** (Baidu) — original ring all-reduce for ML

## Current state (as of project start)

- Mac dev workstation: ssh keys generated, `~/.ssh/config` has `desktop` alias pointing at `personal`, password-less ssh confirmed working
- `personal` (Ubuntu desktop, 1080 Ti): online, `~/tinynccl` git repo initialized, `.gitignore` in place, no code yet
- Laptop (Windows + 4070): Ubuntu install pending — flash drive ordered, install scheduled when it arrives
- Direct ethernet cable: physically connected, IPs not yet configured

## Notes for future AI agents picking this up

- This spec is the source of truth for project scope and sequencing. If reality diverges, update this file rather than letting the README drift.
- Work on `personal` is via ssh from the Mac. The Mac itself never builds or runs anything — it's the editor and orchestrator.
- Default to writing the smallest possible correct version of each milestone, then iterate. Don't preemptively generalize. The spec lays out the trajectory; each commit should advance one step.
- Numerical correctness must be validated at every transport-layer change. A "fast but wrong" all-reduce is worse than no all-reduce at all and will silently break training.
- Performance comparisons against NCCL are the project's deliverable. Save raw numbers in `results/` from day one so progressions are reproducible.
- Owner is a strong applied programmer (quant systems in C++, AI/ML production work) but new to systems-level networking. Explanations should assume comfort with C++ and Python and zero prior knowledge of verbs API, kernel modules, or distributed training internals.
