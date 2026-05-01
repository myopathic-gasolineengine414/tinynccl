# Milestone 0.3: CUDA hello world

Vector addition on the GTX 1080 Ti. Allocate two float arrays on the host, copy them to GPU memory, launch a kernel that sums them element-wise, copy the result back, validate against the host expectation.

## Build and run

```
make run
```

## What this proves

- `nvcc` is installed and the toolchain works against compute capability 6.1 (Pascal, GTX 1080 Ti). When we bring the laptop online, we'll add `-arch=sm_89` for the 4070 (Ada).
- `cudaMalloc` / `cudaMemcpy` / `cudaFree` work end-to-end.
- A CUDA kernel launches with the standard `<<<blocks, threads>>>` syntax and produces correct output.
- Validation: 1,048,576 elements summed in a single kernel launch, each result verified on the CPU.

## What this is not

The simplest possible CUDA program. Single GPU, single stream, default stream, no error recovery beyond a panic macro. Future milestones add complexity one piece at a time.
