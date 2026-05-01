# 3.5: ddp gpt on tinyshakespeare

A ~10.7M-parameter char-level GPT trained on TinyShakespeare across two ranks via DDP through tinynccl. Two CUDA contexts on the same GTX 1080 Ti, gradient sync via softRoCE.

## Architecture

Standard pre-LN decoder-only transformer (nanoGPT-style):

- vocab: 65 unique chars in TinyShakespeare
- block_size: 128
- n_layer: 6, n_head: 6, n_embd: 384
- ~10.7M parameters

## Training run

3000 steps of AdamW (lr 3e-4, weight decay 0.01), batch_size 32, grad clip 1.0. The corpus is split in half across the two ranks.

```
loss 4.344 -> 1.124 (avg of last 50 steps)
3000 steps in 860.6s = 3.5 step/s = 14:20 wall clock
weights consistent across ranks (max diff 0.00e+00)
```

The bottleneck is gradient sync through pinned-host staging — every backward pass moves ~43 MB through `cudaMemcpy → softRoCE → cudaMemcpy` per rank per bucket. On real Mellanox + datacenter GPU + nvidia_peermem the bounce disappears and per-step time would drop substantially.

## Run

```
./run.sh             # verbs, 3000 steps
./run.sh tcp 1000    # tcp transport, 1000 steps
```

Then sample from the saved weights:

```
python3 infer.py "ROMEO:" 500
python3 infer.py "JULIET:" 300
```

## Sample output (after 3000 steps)

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

The model has learned real Shakespeare character names (ROMEO, JULIET, PARIS, GLOUCESTER, CAPULET, TRANIO, CAMILLO, etc.), dialogue structure, archaic vocabulary (thee/thou/wilt), and approximately iambic line shapes. The actual semantics are nonsense but the surface form is unmistakably Shakespearean.

## Bug found and fixed during this milestone

The previous MR registration cache (`fe852ca lib: cache mr registrations`) was unsafe for transient buffers. PyTorch's allocator can recycle a virtual address while remapping it to different physical pages, leaving the cached `ibv_mr` pointing at stale memory. Removed the cache; per-call register/dereg is correct but slower. A future opt-in `register_persistent` API would let stable user buffers benefit from caching while keeping transient buffers safe.
