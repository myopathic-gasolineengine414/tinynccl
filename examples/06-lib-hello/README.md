# 0.6: lib hello

First example that consumes the library at `lib/` instead of writing raw socket / verbs code.

Same shape as 0.1: rank 1 sends a string, rank 0 receives it. The difference is that all the socket setup is hidden behind `tinynccl::Comm::create(...)`. Backend defaults to TCP.

Run:

    make run

Or in two terminals:

    ./hello 0 127.0.0.1
    ./hello 1 127.0.0.1
