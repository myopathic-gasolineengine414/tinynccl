# 3.2: ddp training

Real `torch.nn.parallel.DistributedDataParallel` training with tinynccl as the backend. Two ranks, each with a copy of a small MLP, get different training data, run a few SGD steps, and the gradient sync via our backend keeps both rank's weights identical.

The headline check: after training, `rank 0`'s weights and `rank 1`'s weights match exactly (max diff `0.00e+00`). Different data per rank means different losses per step, but DDP all-reduces gradients before each optimizer step, so the parameter updates are identical.

## Run

```
./run.sh        # tcp transport
./run.sh verbs  # verbs transport
```

## What changed in `lib/torch_backend.cpp`

DDP needs more than just `allreduce`:

- `allgather` / `_allgather_base` — DDP uses these at init to verify parameter shapes match across ranks.
- `broadcast` — used for parameter sync at startup and various optional code paths.
- `barrier` — synchronizes ranks at known points.
- `Work::getFuture()` — DDP chains hooks on the future returned from `allreduce` to do post-reduction work (assigning gradients back, dividing by world_size, etc.). Without this, `loss.backward()` fails inside DDP.

All of these are implemented over the same `Comm::send` / `Comm::recv` / `Comm::all_reduce` primitives the lib already had.

## Limitations

- World size 2.
- CPU float32 only for `allreduce`. `allgather`/`broadcast` are dtype-agnostic at the byte level.
- Static graphs only — we don't implement the bucketing / coalescing optimizations DDP can opportunistically use.
