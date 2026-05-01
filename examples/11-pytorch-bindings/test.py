"""
Phase 3.1: tinynccl bindings for PyTorch tensors.

Two ranks each create a float32 tensor on CPU, fill with rank-specific values,
call our all_reduce, and validate both ranks see the elementwise sum.
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore", message="Failed to initialize NumPy")

import torch
from torch.utils.cpp_extension import load

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LIB_DIR = os.path.join(REPO_ROOT, "lib")

_mod = load(
    name="tinynccl_torch",
    sources=[
        os.path.join(LIB_DIR, "binding.cpp"),
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

def main():
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <rank 0|1> <peer_host> [backend tcp|verbs]")
        sys.exit(1)

    rank = int(sys.argv[1])
    peer = sys.argv[2]
    backend = sys.argv[3] if len(sys.argv) > 3 else "tcp"

    comm = _mod.Comm(
        rank=rank,
        world_size=2,
        peer_host=peer,
        port=5000,
        backend=backend,
    )
    print(f"rank {rank}/{comm.world_size} using {backend}")

    n = 8
    t = torch.zeros(n, dtype=torch.float32)
    t.add_((rank + 1) * 100.0)
    t.add_(torch.arange(n, dtype=torch.float32))

    print(f"rank {rank} before: {t.tolist()}")
    comm.all_reduce(t)
    print(f"rank {rank} after:  {t.tolist()}")

    expected = torch.zeros(n, dtype=torch.float32) + 300.0 + 2.0 * torch.arange(n, dtype=torch.float32)
    if not torch.allclose(t, expected, atol=1e-3):
        print(f"rank {rank}: MISMATCH expected={expected.tolist()}")
        sys.exit(1)
    print(f"rank {rank}: ok")

if __name__ == "__main__":
    main()
