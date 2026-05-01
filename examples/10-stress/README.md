# 10: stress

Correctness + microbenchmark for `Comm::all_reduce`. Runs the op N times with a buffer of K floats, validates every iteration's result against the expected sum, prints end-to-end MB/s.

```
./stress <rank 0|1> <peer_host> <tcp|verbs> [iters=100] [count=262144]
```

Convenience targets:

    make run-tcp     # 100 iters at 256 KB, tcp backend
    make run-verbs   # 100 iters at 256 KB, verbs backend

The verbs backend benefits significantly from MR registration caching (see `lib/src/transport_verbs.cpp`). On softRoCE loopback, latency-dominated regimes (small buffers, many iterations) see ~60x speedup with the cache vs registering per call.

Numbers are not transferable to real RDMA hardware. softRoCE goes through the kernel, not a NIC.
