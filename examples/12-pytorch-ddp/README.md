# 3.1: pytorch c10d backend

Real `c10d::Backend` subclass for tinynccl. After registering it via `torch.distributed.Backend.register_backend("tinynccl", create_backend, devices=["cpu"])`, the standard PyTorch APIs route through our library:

```python
torch.distributed.init_process_group(backend="tinynccl", rank=..., world_size=2, ...)
torch.distributed.all_reduce(tensor, op=ReduceOp.SUM)
```

The backend reads `MASTER_ADDR` / `MASTER_PORT` / `TINYNCCL_BACKEND` from environment for connection info. The c10d `Store` is currently ignored — we use our own TCP for OOB exchange (consistent with the lower-level transports).

## Run

```
./run.sh        # tcp transport
./run.sh verbs  # verbs transport
```

## What this enables

- `torch.distributed.all_reduce(tensor)` works.
- `torch.nn.parallel.DistributedDataParallel(model)` should work with this backend (next milestone tests an actual DDP training loop).
- Anything else PyTorch routes through `Backend::allreduce` works. Other collectives (broadcast, gather, etc.) throw "not supported" — DDP specifically only needs allreduce.

## Limitations

- CPU float32 only.
- `world_size != 2` not supported.
- The `c10d::Store` is ignored. A more idiomatic implementation would use the store for QP info exchange instead of our own TCP.
- No support for the modern `eager_init_pg` / split / coalescing APIs that DDP can opportunistically use.
