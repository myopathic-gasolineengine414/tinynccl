"""
Phase 3.3: DDP training on GPU with tinynccl backend.

Two ranks each share the GTX 1080 Ti via separate CUDA contexts. Each rank
gets its own DDP-wrapped MLP and different training data; gradient sync
flows through tinynccl with pinned-host staging (since softRoCE + consumer
GPU can't do GPUDirect RDMA). After training, weights match across ranks.
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore", message="Failed to initialize NumPy")

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.cpp_extension import load

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LIB_DIR = os.path.join(REPO_ROOT, "lib")

_mod = load(
    name="tinynccl_pg",
    sources=[
        os.path.join(LIB_DIR, "torch_backend.cpp"),
        os.path.join(LIB_DIR, "src", "tinynccl.cpp"),
        os.path.join(LIB_DIR, "src", "transport_tcp.cpp"),
        os.path.join(LIB_DIR, "src", "transport_verbs.cpp"),
    ],
    extra_include_paths=[
        os.path.join(LIB_DIR, "include"),
        os.path.join(LIB_DIR, "src"),
    ],
    extra_ldflags=["-libverbs"],
    verbose=False,
)
dist.Backend.register_backend(
    "tinynccl", _mod.create_backend, devices=["cpu", "cuda"]
)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def main():
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <rank 0|1> <peer_host> [backend tcp|verbs]")
        sys.exit(1)

    rank = int(sys.argv[1])
    peer = sys.argv[2]
    backend = sys.argv[3] if len(sys.argv) > 3 else "verbs"

    os.environ["MASTER_ADDR"] = peer
    os.environ["MASTER_PORT"] = "5000"
    os.environ["TINYNCCL_BACKEND"] = backend

    dist.init_process_group(
        backend="tinynccl",
        rank=rank,
        world_size=2,
        init_method="tcp://127.0.0.1:29500",
    )

    device = torch.device("cuda:0")
    print(f"rank {rank} using {torch.cuda.get_device_name(0)} via {backend}")

    torch.manual_seed(42)
    model = MLP().to(device)
    ddp = DDP(model, device_ids=[0])
    opt = torch.optim.SGD(ddp.parameters(), lr=0.05)

    torch.manual_seed(100 + rank)
    x = torch.randn(16, 10, device=device)
    y = torch.randn(16, 1, device=device)

    for step in range(5):
        opt.zero_grad()
        out = ddp(x)
        loss = ((out - y) ** 2).mean()
        loss.backward()
        opt.step()
        print(f"rank {rank} step {step}: loss={loss.item():.4f}")

    # Verify weights match across ranks (DDP guarantees this).
    fc1_w = model.fc1.weight.data.clone()
    summed = fc1_w.clone()
    dist.all_reduce(summed, op=dist.ReduceOp.SUM)
    summed /= 2.0
    diff = (summed - fc1_w).abs().max().item()
    if diff < 1e-5:
        print(f"rank {rank}: weights consistent across ranks (max diff {diff:.2e})")
    else:
        print(f"rank {rank}: WEIGHTS DIVERGED (max diff {diff:.2e})")
        sys.exit(1)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
