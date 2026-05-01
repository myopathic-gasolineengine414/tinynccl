# 0.1: tcp echo

Two-process TCP echo on a single machine. Server listens on port 5000, client sends a string, server echoes it back, server exits.

Here mainly to confirm the build pipeline works and to baseline the BSD socket API before swapping it out for verbs in 0.2.

Run:

    make run

Or in two terminals:

    ./server
    ./client [host] [port] [message]
