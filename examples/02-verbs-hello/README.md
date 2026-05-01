# 0.2: verbs hello

Single-machine RDMA hello-world over softRoCE (rxe0). Same client/server skeleton as 0.1, but `send` / `recv` are replaced with `ibv_post_send` / `ibv_post_recv` over a verbs QP.

The TCP socket is still around — just for OOB exchange of QP metadata (QPN, PSN, GID). The actual data transfer goes over verbs.

What gets exercised: device open, PD, MR, CQ, QP creation, the QP state machine (INIT → RTR → RTS), post send/recv, CQ poll. The rxe device is software, so packets don't actually leave the kernel — but the API path is identical to what runs on real Mellanox hardware.

Run:

    make run
