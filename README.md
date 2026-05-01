# tinynccl

A from-scratch C++ implementation of GPU collective communication built on libibverbs over softRoCE, exposed as a custom PyTorch `c10d::Backend`. Used to train a real model end-to-end via `DistributedDataParallel` with the gradient sync flowing through code in this repo.

```
ROMEO:
I fear thee bear thee to undertake thee:
Provost, thou wilt not let me come to him:
Do wear the moral duke of thy friends to tear:
...

PARIS:
How now! what news?

POMPEY:
My lord, I come, sir; and I once a poor true.
```

Sample output from a 10.7M-param char-level GPT trained on TinyShakespeare across two GPU ranks via `DistributedDataParallel`, with gradient all-reduce flowing through `tinynccl::Comm` over libibverbs / softRoCE / `rdma_rxe`. Loss converged 4.34 to 1.12 over 3000 steps in 14:20 on a single GTX 1080 Ti (two CUDA contexts).

## What this is

A library that lets PyTorch's standard distributed APIs route through code I wrote, all the way down to the verbs API:

```
torch.distributed.all_reduce(tensor)
  -> c10d::Backend::allreduce      (lib/torch_backend.cpp)
    -> tinynccl::Comm::all_reduce  (lib/src/tinynccl.cpp)
      -> Transport::send / recv    (lib/src/transport_verbs.cpp)
        -> ibv_post_send / poll_cq (libibverbs over softRoCE)
          -> kernel rdma_rxe
```

Two transport backends are pluggable (TCP for baseline, verbs for the production path). Both implement the same `Transport` interface and the higher layers don't care which is in use.

Built as a learning artifact for an NVIDIA Networking Systems Software Engineering internship application. The pitch is not "this competes with NCCL" (it doesn't, by orders of magnitude). The pitch is "I have walked every layer of the stack with working code and can talk about any layer in an interview."

## What this is not

Honesty matters more than marketing for this kind of project, so:

- **softRoCE is software emulation, not real RDMA hardware.** Performance numbers from this repo do not transfer to a real Mellanox ConnectX. The API path is identical, the perf is not.
- **The 1080 Ti is a consumer GeForce, which historically does not support GPUDirect RDMA.** Strict-mode `ibv_reg_mr` on a `cudaMalloc`'d device pointer fails with `EFAULT` (the repo demonstrates this in `examples/04-cuda-verbs/`). The library handles it via a pinned-host staging path (the same fallback production HPC code uses when GPUDirect is not available). On real Mellanox + datacenter GPU + `nvidia_peermem` the bounce disappears.
- **No multi-machine support yet.** All examples run on a single Linux box with two CUDA contexts on one GPU. The cross-machine work (Phase 1 in `SPEC.md`) is blocked on hardware.
- **No NCCL benchmark.** A credible comparison requires multi-machine and ideally a real NIC; deferred until Phase 1 lands.
- **2 ranks only.** The library hardcodes `world_size == 2`. Generalizing to N ranks is a milestone, not a one-line change.
- **CPU + CUDA, float32, sum reduction.** All it takes to demonstrate the stack works; doesn't aim to be feature-complete.

## Quick start

```bash
# Prerequisites on Ubuntu 24.04:
sudo apt install -y build-essential libibverbs-dev rdma-core ibverbs-utils
sudo modprobe rdma_rxe
sudo rdma link add rxe0 type rxe netdev <your-iface>   # e.g. wlp10s0

# Build the library
cd lib && make

# Smallest end-to-end example: TCP echo (no GPU, no RDMA)
cd ../examples/01-tcp-echo && make run

# Smallest RDMA example: libibverbs hello-world over softRoCE
cd ../02-verbs-hello && make run

# Library version of all-reduce over verbs
cd ../08-allreduce-verbs && make run
```

For the PyTorch / DDP examples, set up a venv and install torch:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install ninja torch --index-url https://download.pytorch.org/whl/cu121

# Sine fit (small model, runs in seconds)
cd examples/15-ddp-sine && ./run.sh

# GPT training (10.7M params, ~15 min on a 1080 Ti)
cd examples/16-ddp-gpt && ./run.sh
```

## Repository layout

```
lib/                          the actual library
├── include/tinynccl.h        public C++ API
├── src/transport.h           Transport interface
├── src/transport_tcp.cpp     TCP backend
├── src/transport_verbs.cpp   verbs backend (libibverbs over softRoCE)
├── src/tinynccl.cpp          Comm + all_reduce algorithm
├── binding.cpp               pybind11 wrapper for raw Comm
└── torch_backend.cpp         c10d::Backend subclass for torch.distributed

examples/
├── 01-tcp-echo               sockets warm-up
├── 02-verbs-hello            libibverbs SEND/RECV over rxe0
├── 03-cuda-vec-add           CUDA toolchain check
├── 04-cuda-verbs             GPU+verbs SEND/RECV with documented GPUDirect fallback
├── 05-rdma-write             one-sided RDMA WRITE_WITH_IMM
├── 06-lib-hello              first example using the library
├── 07-allreduce-tcp          2-rank all-reduce over TCP
├── 08-allreduce-verbs        2-rank all-reduce over verbs
├── 09-allreduce-gpu          GPU all-reduce on one GPU, two CUDA contexts
├── 10-stress                 correctness + microbenchmark for repeated all-reduce
├── 11-pytorch-bindings       PyTorch tensors through tinynccl (precursor to backend)
├── 12-pytorch-ddp            torch.distributed.init_process_group(backend="tinynccl")
├── 13-ddp-train              MLP via DDP, weights consistent across ranks
├── 14-ddp-gpu                MLP via DDP on CUDA tensors
├── 15-ddp-sine               first non-trivial training task: fit y = sin(2*pi*x)
└── 16-ddp-gpt                10.7M-param char-level GPT on TinyShakespeare
```

Each example directory is self-contained with its own README and Makefile. The numbered prefix is roughly the order they were built; later examples depend on the library scaffolded in earlier ones.

## SPEC.md

`SPEC.md` in the repo root is the source of truth for project scope, current status, hardware setup, milestones (done and planned), and known limitations. Read it for technical depth.

## A real bug found and fixed during this project

The verbs transport originally cached `ibv_reg_mr` results by buffer address to amortize registration cost (it gave a 64x speedup at small messages). When integrated with PyTorch, the cache turned out to be unsafe: PyTorch's allocator can recycle a virtual address while remapping the underlying physical pages, leaving cached `ibv_mr` handles pointing at stale memory. The symptom was DDP's parameter-shape verification reporting phantom mismatches because broadcast was returning data from the wrong physical pages.

Fix: register-and-deregister per call (slower but correct). Logged future work: an opt-in `register_persistent` API for caller-owned stable buffers to recover the perf benefit safely.

This is documented honestly in `lib/src/transport_verbs.cpp` and the relevant commits (`fe852ca` introduced the cache, `ff8781c` removed it).

## What I would change on real hardware

- `ibv_reg_mr` on `cudaMalloc`'d device pointers would succeed (with `nvidia_peermem` loaded), eliminating the pinned-host staging bounce.
- The verbs path would be 10-100x faster at the wire layer (real DMA vs kernel software path).
- Per-step latency of DDP training would drop substantially because the gradient sync is the dominant cost on this hardware.
- The N-rank generalization (ring all-reduce) would matter, since 2 ranks is the simplest case where ring degenerates to direct exchange.

The library code is structured so these changes are scoped (verbs registration path, transport bandwidth, algorithm choice) rather than requiring a rewrite.
