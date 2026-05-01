# 3.4: ddp sine fit (a model that actually learns)

Earlier DDP examples (3.2, 3.3) trained on `torch.randn → torch.randn` — pure noise. They validated DDP gradient sync mechanics through tinynccl, but there was no signal to learn.

This one fits `y = sin(2πx)` across two ranks, each with half the training data, on the same GTX 1080 Ti. Loss drops from ~0.21 to ~0.016 over 200 Adam steps. Inference on a held-out grid produces predictions that track the sine curve.

## Run

```
./run.sh        # verbs (default)
./run.sh tcp
```

`run.sh` trains the model, saves `model.pt`, then runs `infer.py` standalone to confirm the saved weights work outside the distributed training run.

## Why this is the first "real" training milestone

Every component is non-trivial:

- **Real loss curve**: 0.21 → 0.016, monotonic decrease.
- **Real inference**: `infer.py` loads the weights and produces predictions matching the target function.
- **Real DDP**: gradient sync via `dist.all_reduce` through tinynccl over softRoCE.
- **Real cross-rank consistency**: weights match between ranks at machine precision after every optimizer step.

The model is small and the task is small, but the full pipeline works the way a real distributed training run works.

## Limitations

- The MLP underfits at the boundaries (predicts ±0.27 at x=0/1 instead of 0). More steps or a wider net would fix this. We trained briefly to keep run time short.
- Single GPU shared between ranks. With two physical GPUs on a real network, perf would be very different.
- Adam moments are not synced via DDP across ranks — only gradients are. This is the standard DDP behavior.
