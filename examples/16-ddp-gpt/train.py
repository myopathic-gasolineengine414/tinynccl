"""
Phase 3.5: train a mini char-level GPT on TinyShakespeare via DDP through tinynccl.

~10M parameters. Two ranks share the same GTX 1080 Ti (separate CUDA contexts).
Each rank gets a disjoint half of the corpus. After training, rank 0 generates
sample text from a prompt and saves the weights.

Architecture: standard pre-LN decoder-only transformer.
"""
import math
import os
import sys
import time
import urllib.request
import warnings
warnings.filterwarnings("ignore", message="Failed to initialize NumPy")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.cpp_extension import load

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LIB_DIR = os.path.join(REPO_ROOT, "lib")
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "tinyshakespeare.txt")
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

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


# ~10M params at vocab=65
BLOCK_SIZE = 128
N_LAYER = 6
N_HEAD = 6
N_EMBD = 384


def get_data():
    if not os.path.exists(DATA_PATH):
        print(f"downloading TinyShakespeare to {DATA_PATH}")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    with open(DATA_PATH, "r") as f:
        return f.read()


class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(N_EMBD, 3 * N_EMBD, bias=False)
        self.proj = nn.Linear(N_EMBD, N_EMBD, bias=False)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)).view(1, 1, BLOCK_SIZE, BLOCK_SIZE),
        )

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(N_EMBD, dim=2)
        head_dim = C // N_HEAD
        q = q.view(B, T, N_HEAD, head_dim).transpose(1, 2)
        k = k.view(B, T, N_HEAD, head_dim).transpose(1, 2)
        v = v.view(B, T, N_HEAD, head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.attn = CausalSelfAttention()
        self.ln2 = nn.LayerNorm(N_EMBD)
        self.mlp = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD),
            nn.GELU(),
            nn.Linear(4 * N_EMBD, N_EMBD),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.tok_emb = nn.Embedding(vocab_size, N_EMBD)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(*[Block() for _ in range(N_LAYER)])
        self.ln = nn.LayerNorm(N_EMBD)
        self.head = nn.Linear(N_EMBD, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8):
        for _ in range(max_new_tokens):
            ctx = idx[:, -BLOCK_SIZE:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


def main():
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <rank 0|1> <peer_host> [backend tcp|verbs] [steps=3000]")
        sys.exit(1)

    rank = int(sys.argv[1])
    peer = sys.argv[2]
    backend = sys.argv[3] if len(sys.argv) > 3 else "verbs"
    steps = int(sys.argv[4]) if len(sys.argv) > 4 else 3000

    os.environ["MASTER_ADDR"] = peer
    os.environ["MASTER_PORT"] = "5000"
    os.environ["TINYNCCL_BACKEND"] = backend

    dist.init_process_group(
        backend="tinynccl", rank=rank, world_size=2,
        init_method="tcp://127.0.0.1:29500",
    )

    device = torch.device("cuda:0")

    text = get_data()
    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

    half = len(data) // 2
    shard = data[:half] if rank == 0 else data[half:]

    torch.manual_seed(42)
    model = GPT(vocab_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    ddp = DDP(model, device_ids=[0])
    opt = torch.optim.AdamW(ddp.parameters(), lr=3e-4, weight_decay=0.01)

    print(f"rank {rank}: device={torch.cuda.get_device_name(0)}, backend={backend}, "
          f"vocab={vocab_size}, params={n_params:,} ({n_params / 1e6:.1f}M), "
          f"shard={len(shard):,} chars, steps={steps}", flush=True)

    batch_size = 32
    t0 = time.time()
    losses = []
    for step in range(steps):
        ix = torch.randint(0, len(shard) - BLOCK_SIZE - 1, (batch_size,))
        xb = torch.stack([shard[i:i + BLOCK_SIZE] for i in ix]).to(device)
        yb = torch.stack([shard[i + 1:i + BLOCK_SIZE + 1] for i in ix]).to(device)
        opt.zero_grad()
        _, loss = ddp(xb, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ddp.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        if (step + 1) % 50 == 0:
            avg = sum(losses[-50:]) / 50
            elapsed = time.time() - t0
            sps = (step + 1) / elapsed
            eta = (steps - step - 1) / sps
            print(f"rank {rank} step {step + 1:4d}/{steps}: loss={avg:.4f}  "
                  f"({sps:.1f} step/s, ETA {eta:.0f}s)", flush=True)

    elapsed = time.time() - t0
    print(f"rank {rank}: trained {steps} steps in {elapsed:.1f}s "
          f"(loss {losses[0]:.3f} -> avg-last-50 {sum(losses[-50:])/50:.3f})", flush=True)

    # Cross-rank weight check.
    w = list(model.parameters())[0].data.clone()
    s = w.clone()
    dist.all_reduce(s, op=dist.ReduceOp.SUM)
    s /= 2.0
    diff = (s - w).abs().max().item()
    print(f"rank {rank}: weights consistent across ranks (max diff {diff:.2e})", flush=True)

    if rank == 0:
        model.eval()
        for prompt in ["ROMEO:", "JULIET:", "First Citizen:"]:
            idx = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long, device=device)
            out = model.generate(idx, max_new_tokens=300)[0].tolist()
            text_out = "".join(itos[i] for i in out)
            print()
            print("=" * 60)
            print(f"sample (prompt = {prompt!r}):")
            print(text_out)
            print("=" * 60)

        out_path = os.path.join(HERE, "model.pt")
        torch.save({"state_dict": model.state_dict(), "vocab": chars}, out_path)
        print(f"\nsaved model to {out_path}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
