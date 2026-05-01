"""
Generate samples from the GPT trained by train.py. No distributed setup —
just load weights and sample.
"""
import math
import os
import sys
import warnings
warnings.filterwarnings("ignore", message="Failed to initialize NumPy")

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))

BLOCK_SIZE = 128
N_LAYER = 6
N_HEAD = 6
N_EMBD = 384


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
        return self.proj((att @ v).transpose(1, 2).contiguous().view(B, T, C))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.attn = CausalSelfAttention()
        self.ln2 = nn.LayerNorm(N_EMBD)
        self.mlp = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD), nn.GELU(),
            nn.Linear(4 * N_EMBD, N_EMBD),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, N_EMBD)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(*[Block() for _ in range(N_LAYER)])
        self.ln = nn.LayerNorm(N_EMBD)
        self.head = nn.Linear(N_EMBD, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.blocks(x)
        return self.head(self.ln(x))

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8):
        for _ in range(max_new_tokens):
            ctx = idx[:, -BLOCK_SIZE:]
            logits = self(ctx)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


def main():
    weights_path = os.path.join(HERE, "model.pt")
    if not os.path.exists(weights_path):
        print(f"no weights at {weights_path}; run train.py first")
        sys.exit(1)

    ckpt = torch.load(weights_path, map_location="cpu", weights_only=True)
    chars = ckpt["vocab"]
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = GPT(len(chars)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    prompt = sys.argv[1] if len(sys.argv) > 1 else "ROMEO:"
    n_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 500

    idx = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=n_tokens)[0].tolist()
    print("".join(itos[i] for i in out))


if __name__ == "__main__":
    main()
