"""
Phase 3.4: real DDP training that actually learns something.

Fit a small MLP to y = sin(2*pi*x) across two ranks on the same GPU via
tinynccl. Each rank gets a different shard of the training data. After
training, both ranks evaluate on a held-out grid and print predictions
next to ground truth.
"""
import os
import sys
import math
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
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


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
        backend="tinynccl", rank=rank, world_size=2,
        init_method="tcp://127.0.0.1:29500",
    )

    device = torch.device("cuda:0")

    # Dataset: 1024 points (x, sin(2*pi*x)) split across the 2 ranks.
    torch.manual_seed(0)
    n_total = 1024
    x_all = torch.rand(n_total, 1)
    y_all = torch.sin(2 * math.pi * x_all)
    half = n_total // 2
    if rank == 0:
        x_train, y_train = x_all[:half].to(device), y_all[:half].to(device)
    else:
        x_train, y_train = x_all[half:].to(device), y_all[half:].to(device)

    # Both ranks start with identical weights so DDP has something to keep in sync.
    torch.manual_seed(42)
    model = MLP(hidden=64).to(device)
    ddp = DDP(model, device_ids=[0])
    opt = torch.optim.Adam(ddp.parameters(), lr=5e-3)

    print(f"rank {rank}: device={torch.cuda.get_device_name(0)}, backend={backend}, "
          f"train_n={x_train.shape[0]}")

    batch_size = 64
    steps = 200
    losses = []
    for step in range(steps):
        idx = torch.randint(0, x_train.shape[0], (batch_size,), device=device)
        xb, yb = x_train[idx], y_train[idx]
        opt.zero_grad()
        loss = ((ddp(xb) - yb) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if (step + 1) % 25 == 0:
            print(f"rank {rank} step {step + 1:4d}: loss={loss.item():.5f}")

    # Sanity: weights match across ranks.
    fc = model.net[0].weight.data.clone()
    fc_summed = fc.clone()
    dist.all_reduce(fc_summed, op=dist.ReduceOp.SUM)
    fc_summed /= 2.0
    diff = (fc_summed - fc).abs().max().item()
    print(f"rank {rank}: weights consistent across ranks (max diff {diff:.2e})")

    # Inference on a held-out grid.
    if rank == 0:
        model.eval()
        xs = torch.linspace(0, 1, 11, device=device).unsqueeze(1)
        with torch.no_grad():
            ys = model(xs).squeeze().cpu()
        truth = torch.sin(2 * math.pi * xs.squeeze().cpu())
        print()
        print("inference (rank 0):")
        print(f"  {'x':>5} {'pred':>10} {'sin(2pi*x)':>12} {'err':>10}")
        for x, p, t in zip(xs.squeeze().cpu(), ys, truth):
            print(f"  {x.item():.2f}  {p.item():>+9.4f}    {t.item():>+9.4f}  "
                  f"{abs(p.item() - t.item()):>+9.4f}")
        rmse = ((ys - truth) ** 2).mean().sqrt().item()
        print(f"\n  rmse on test grid: {rmse:.4f}")

        # Save weights so we can reload and do inference outside the training run.
        out_path = os.path.join(os.path.dirname(__file__), "model.pt")
        torch.save(model.state_dict(), out_path)
        print(f"  saved weights to {out_path}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
