# 0.3: cuda vec_add

Vector addition on the GTX 1080 Ti. Two float arrays on the host, copy to device, kernel sums them, copy back, validate. ~50 LOC.

Compile flags target sm_61 (Pascal). Once the laptop comes online we'll add sm_89 for the 4070.

Run:

    make run
