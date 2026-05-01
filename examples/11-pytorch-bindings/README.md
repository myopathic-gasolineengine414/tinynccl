# 3.0: pytorch bindings

Pybind11 wrapper that lets `tinynccl::Comm` accept `torch::Tensor` directly. Two PyTorch processes each create a CPU float32 tensor, call `comm.all_reduce(tensor)`, and both end with the elementwise sum.

This is a precursor to a real `c10d::Backend` subclass (Phase 3.1). The full Backend would let `torch.distributed.all_reduce` and DDP route through tinynccl. Here we just expose the library to Python and accept tensors as buffers — sufficient to demonstrate PyTorch integration end-to-end without the full `Backend` API surface.

## Run

```
./run.sh        # tcp backend
./run.sh verbs  # verbs backend
```

The first run takes ~30-60s because `torch.utils.cpp_extension.load` compiles `lib/binding.cpp` together with the lib sources. Subsequent runs are cached at `~/.cache/torch_extensions/` and take seconds.

## What this does NOT do (yet)

- Not a `c10d::Backend`. `torch.distributed.init_process_group(backend="tinynccl")` does not work.
- Not GPU-aware. Tensor must be on CPU and float32.
- No DDP integration. You can't drop this into `DistributedDataParallel` yet.

Phase 3.1 will turn this into a proper backend by subclassing `c10d::Backend` and registering it with `torch.distributed`.

## Why this still matters

It proves the C++/Python boundary works: PyTorch tensors flow through tinynccl correctly, both backends carry the data, numerical results match expectations. The remaining Backend wrapping is mechanical glue.
