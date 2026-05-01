# Milestone 0.2: libibverbs hello-world

Single-machine RDMA hello-world over softRoCE (rxe0). Same two-process structure as `01-tcp-echo`, but the data path is now verbs: client posts a SEND, server posts a RECV, both poll their completion queues. Every layer of the verbs stack is exercised — device, PD, MR, CQ, QP, the QP state machine (INIT → RTR → RTS), and post/poll.

A TCP socket is still here, used **only for out-of-band exchange of QP metadata** (QPN, PSN, GID). After exchange, the actual data transfer is verbs.

## Build and run

```
make run
```

Or in two terminals (default port 5000):

```
# terminal 1
./server

# terminal 2
./client
```

## What this proves

- libibverbs userland works against rxe0: `ibv_open_device`, `ibv_alloc_pd`, `ibv_reg_mr`, `ibv_create_cq`, `ibv_create_qp` all succeed.
- QP state machine works: INIT → RTR → RTS transitions complete cleanly.
- Data plane works: a SEND on the client lands at a pre-posted RECV on the server, and the server's CQ surfaces a successful WC with `byte_len` matching what was sent.
- The OOB exchange pattern (TCP for setup, verbs for data) is in place. Every later milestone reuses this skeleton.

## What this is not

Single-machine, single-message, one-shot. No connection management, no error recovery beyond per-call return checks, no batching, no flow control beyond what the QP gives you. The point is to validate the API works end-to-end.

## Concept cheat sheet

- **PD (Protection Domain)** — like a process group; MRs and QPs allocated against one PD can interact, those in different PDs cannot.
- **MR (Memory Region)** — registered region of process memory the NIC can read/write. `ibv_reg_mr` pins the pages and returns local/remote keys (`lkey`, `rkey`).
- **CQ (Completion Queue)** — ring buffer of WC entries. Verbs is async; you poll the CQ to find out when posted operations finished.
- **QP (Queue Pair)** — pair of work queues (send + recv) plus connection state. Type RC = Reliable Connection, the InfiniBand analogue of TCP semantics.
- **WR (Work Request)** — a job posted to a queue: SEND, RECV, RDMA_WRITE, RDMA_READ, ATOMIC.
- **WC (Work Completion)** — entry on the CQ describing a completed WR: `status`, `byte_len`, `wr_id`.
- **SGE (Scatter/Gather Element)** — pointer + length + lkey. A WR carries an array of these.
- **GID (Global IDentifier)** — IPv6-format address used for routing. Required for RoCE; LID is typically 0.
- **PSN (Packet Sequence Number)** — 24-bit; both sides exchange initial values for ordering.
