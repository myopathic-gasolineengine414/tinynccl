"""
Phase 3.1: register tinynccl as a c10d Backend so torch.distributed.all_reduce
routes through our library.
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore", message="Failed to initialize NumPy")

import torch
import torch.distributed as dist
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

    # init_process_group needs a store, even if our backend ignores it.
    # We use TCPStore on a different port for that side-channel.
    dist.init_process_group(
        backend="tinynccl",
        rank=rank,
        world_size=2,
        init_method="tcp://127.0.0.1:29500",
    )

    print(f"rank {rank}/{dist.get_world_size()} using {backend}")

    n = 8
    t = torch.zeros(n, dtype=torch.float32)
    t.add_((rank + 1) * 100.0)
    t.add_(torch.arange(n, dtype=torch.float32))
    print(f"rank {rank} before: {t.tolist()}")

    dist.all_reduce(t, op=dist.ReduceOp.SUM)

    print(f"rank {rank} after:  {t.tolist()}")

    expected = torch.zeros(n, dtype=torch.float32) + 300.0 + 2.0 * torch.arange(n, dtype=torch.float32)
    if not torch.allclose(t, expected, atol=1e-3):
        print(f"rank {rank}: MISMATCH expected={expected.tolist()}")
        sys.exit(1)
    print(f"rank {rank}: ok")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
