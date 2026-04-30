# tinynccl

A from-scratch C++/CUDA implementation of GPU collective communication, built on libibverbs over softRoCE, integrated with PyTorch as a custom `ProcessGroup` backend. Trains a real model end-to-end across two physical machines using transport code I wrote myself.

**Status:** in progress. See [SPEC.md](SPEC.md) for full project scope, milestones, and current state.

## What it does

- Implements all-reduce (and friends) over libibverbs / RDMA, with a TCP fallback for sanity comparison
- Registers GPU memory directly with the network card so transfers don't bounce through host RAM
- Plugs into PyTorch as a `ProcessGroup` backend so standard DDP training routes through it transparently
- Includes a benchmark suite comparing against real NCCL

## Why

Built as a learning artifact for an NVIDIA Networking Systems Software Engineering internship application. The goal is to walk every layer from `torch.distributed.all_reduce` down to `ibv_post_send`, end-to-end, on real hardware, and produce honest numbers.

## Hardware tested on

- Two Linux boxes (GTX 1080 Ti + RTX 4070) connected by a single ethernet cable
- softRoCE (kernel `rdma_rxe`) standing in for hardware RDMA NICs
- See [SPEC.md](SPEC.md) for the full topology

## Building and running

(This section will fill in as milestones land. See [SPEC.md](SPEC.md) for current state.)
