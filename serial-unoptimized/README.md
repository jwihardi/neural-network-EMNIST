# serial-unoptimized

> **Tools:** plain C++20, one thread, no libraries.

The baseline. Trains one sample at a time: forward/backward is a chain of matrix-vector products written as naive nested loops.

## Why it's slow

Everything comes down to one number: the weights are re-read and re-written **once per sample**.

- Forward does `W1 (hidden×784) · x (784)` — all of `W1` streams through the core to produce one column of output. There is no reuse: each weight is loaded, used for a single multiply-add, and evicted.
- The gradient step `W -= lr · (dz ⊗ x)` writes **every element of both weight matrices for every image**. On `digits` that's 240k full read-modify-write passes over the weights per epoch.
- The dot-product inner loops walk columns with a stride, so the compiler can't vectorize them well either.

The FLOP count is identical to every other variant — the machine just spends nearly all its time moving the same bytes through the cache hierarchy over and over. ~15 s for one epoch of `digits` (240k images); the same epoch takes ~0.1 s on the GPU.

## What it's for

- The readable reference: the math maps 1:1 onto the code, and the other four variants are checked against its results.
- CLI is `<dataset> <epochs> <lr> <hidden>` — no batch args, because there are no batches.

## Next rung

[serial-optimized](../serial-optimized/README.md) fixes exactly one thing — amortizing the weight traffic over a batch — and gets ~2.5× on the same single core.
