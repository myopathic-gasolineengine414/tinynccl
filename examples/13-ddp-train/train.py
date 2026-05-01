"""
Phase 3.2: real DDP training using tinynccl as the backend.

Two ranks each have a copy of a small MLP. Each rank gets different
training data. After a few SGD steps via DistributedDataParallel,
the two model copies must have identical weights (which is what DDP
guarantees — gradients are all-reduced before each optimizer step).
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
dist.Backend.register_backend("tinynccl", _mod.create_backend, devices=["cpu"])


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
    backend = sys.argv[3] if len(sys.argv) > 3 else "tcp"

    os.environ["MASTER_ADDR"] = peer
    os.environ["MASTER_PORT"] = "5000"
    os.environ["TINYNCCL_BACKEND"] = backend

    dist.init_process_group(
        backend="tinynccl",
        rank=rank,
        world_size=2,
        init_method="tcp://127.0.0.1:29500",
    )

    torch.manual_seed(42)  # both ranks start with identical weights
    model = MLP()
    ddp = DDP(model)
    opt = torch.optim.SGD(ddp.parameters(), lr=0.05)

    # Different data per rank — without gradient sync the models would diverge.
    torch.manual_seed(100 + rank)
    x = torch.randn(16, 10)
    y = torch.randn(16, 1)

    for step in range(5):
        opt.zero_grad()
        out = ddp(x)
        loss = ((out - y) ** 2).mean()
        loss.backward()
        opt.step()
        print(f"rank {rank} step {step}: loss={loss.item():.4f}")

    # After training, both ranks must have identical weights.
    fc1_w = model.fc1.weight.data.clone()
    if rank == 0:
        # rank 0 sends its weights, rank 1 receives and compares.
        # Use the standard distributed primitive.
        dist.all_reduce(fc1_w, op=dist.ReduceOp.SUM)
        # All-reducing identical tensors yields 2x. Divide.
        fc1_w /= 2.0
        # Now compare with our local copy.
        if torch.allclose(fc1_w, model.fc1.weight.data, atol=1e-5):
            print(f"rank {rank}: weights consistent across ranks (max diff "
                  f"{(fc1_w - model.fc1.weight.data).abs().max().item():.2e})")
        else:
            print(f"rank {rank}: WEIGHTS DIVERGED — DDP gradient sync broken")
            sys.exit(1)
    else:
        dist.all_reduce(fc1_w, op=dist.ReduceOp.SUM)
        fc1_w /= 2.0
        if torch.allclose(fc1_w, model.fc1.weight.data, atol=1e-5):
            print(f"rank {rank}: weights consistent")
        else:
            print(f"rank {rank}: WEIGHTS DIVERGED")
            sys.exit(1)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
