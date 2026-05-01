# 2.1: allreduce-tcp

Naive 2-rank in-place all-reduce over the TCP backend. Each rank fills a float array with rank-specific values, calls `comm->all_reduce(...)`, and after the call both arrays should hold the elementwise sum.

Algorithm: rank 0 sends its buffer, rank 1 receives into a temp; rank 1 sends its buffer, rank 0 receives into a temp; both add the temp into their local buffer. The send/recv ordering is alternated to avoid deadlock on the synchronous transport.

Run:

    make run

Or in two terminals:

    ./allreduce 0 127.0.0.1
    ./allreduce 1 127.0.0.1
