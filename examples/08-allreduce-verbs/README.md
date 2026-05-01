# 2.2: allreduce-verbs

Same as `07-allreduce-tcp` but with `Backend::Verbs` — bytes flow over rxe0 instead of TCP.

The `Comm::create(...)` line is the only difference. The all-reduce algorithm in `lib/` is unchanged. This validates the transport abstraction.

Run:

    make run

The library buffers passed to `send`/`recv` get registered on the fly via `ibv_reg_mr` and deregistered after each call. This is the slow path; future work can add buffer registration caching for hot paths.
