# 0.5: rdma write

One-sided `RDMA_WRITE_WITH_IMM`. Client reaches into server's memory directly. Server CPU doesn't participate in the data transfer.

How it differs from 0.2 (SEND/RECV):
- Server publishes its buffer's `addr` and `rkey` via OOB exchange (extended `qp_info`).
- Client posts an `IBV_WR_RDMA_WRITE_WITH_IMM` work request with `wr.rdma.remote_addr` set to the server's `addr` and `wr.rdma.rkey` set to its `rkey`.
- The data lands at server's buffer with no RECV needed for the payload.
- The `_WITH_IMM` variant carries 4 bytes of immediate data and consumes one RECV WQE on the server, generating a CQE so the server knows the write happened. Without IMM, the server would have to poll memory or use an out-of-band signal to detect the write.

The server's MR has `IBV_ACCESS_REMOTE_WRITE` set, which is what makes the rkey valid for remote writes.

Run:

    make run

This is the operation NCCL and MPI use for bulk data. SEND/RECV is for control messages.
