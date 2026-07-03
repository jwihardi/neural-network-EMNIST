# serial-optimized

> **Tools:** plain C++20, one thread, no libraries — identical compiler flags to the unoptimized build; the diff is purely code.

Same math as [serial-unoptimized](../serial-unoptimized/README.md), restructured around mini-batches: activations become `hidden × batch` matrices and every op is a real GEMM.

## Why it beats unoptimized

It attacks the exact problem the baseline has — weight traffic — without adding a single thread:

- **Weights load once per batch, not once per sample.** Each `W` element loaded into a register now does `batch` multiply-adds instead of 1. Arithmetic intensity scales linearly with batch size, so the loops flip from memory-bound toward compute-bound.
- **The gradient write-storm collapses.** `W -= lr/B · (dZ · Xᵀ)` updates the weights once per batch — on `digits` at batch 128 that's 1,875 weight writes per epoch instead of 240,000.
- **The inner loop vectorizes.** Loop order is `(i, u, b)` with the batch dimension innermost: writes are contiguous, the same weight is broadcast across a SIMD register, and `__restrict` tells the compiler nothing aliases — this is what unlocks AVX-512 FMA.

Result: ~2.5× over the baseline on one core (5.9 s vs 14.7 s, `digits` 1 epoch).

## What still limits it

One core. The loops are now shaped exactly like a textbook GEMM — which means a BLAS library or a GPU can run the same structure far better. There's also no cache blocking: for large `hidden × batch` tiles the operands stop fitting in L2 and it drifts back toward memory-bound.

## Next rung

Three different answers to "one core":
[parallel-omp](../parallel-omp/README.md) throws threads at these exact loops, [parallel-cblas](../parallel-cblas/README.md) replaces them with expert-written ones, [parallel-cuda](../parallel-cuda/README.md) changes the hardware.
