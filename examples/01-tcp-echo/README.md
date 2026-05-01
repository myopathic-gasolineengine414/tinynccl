# Milestone 0.1: TCP echo

Two-process TCP echo on a single machine. Server listens, client sends a string, server echoes back, client prints it. Server then exits.

## Build and run

```
make run
```

Or in two separate terminals:

```
# terminal 1
./server

# terminal 2
./client
```

`./client <host> <port> <message>` overrides the defaults (127.0.0.1, 5000, "hello, tinynccl").

## What this proves

- Build pipeline (g++, Makefile) works end-to-end on `personal`.
- BSD sockets API works: `socket`, `bind`, `listen`, `accept`, `connect`, `send`, `recv`.
- The server-client two-process model on one box is functional, which is the structural skeleton every later milestone reuses.

## What this is not

This is intentionally the smallest possible thing. No threading, no error recovery beyond perror, no graceful shutdown, no message framing, single connection then exit. Each later milestone adds one new concept to this baseline.

The next milestone (`02-verbs-hello`) replaces `send`/`recv` with `ibv_post_send`/`ibv_post_recv` over softRoCE, keeping the surrounding two-process structure identical.
