# serial-unoptimized

> **Tools:** plain C++20, one thread, no libraries.

The baseline. Trains one sample at a time — every "matrix op" is a naive nested loop over a single column, so forward/backward is a chain of vector-times-matrix operations with no data reuse.

## Performance
Slowest by orders of magnitude. Per-sample SGD means the weight matrices are re-read from memory for every single image, so it is memory-bound with zero arithmetic intensity. ~15 s for one epoch of `digits` (240k images) where the GPU version takes ~0.1 s.

## Upsides
- Easiest to read; the math maps 1:1 onto the code.
- The reference the other four implementations are checked against.

## Downsides
- No batching, so no cache reuse, no vectorization-friendly inner loops, nothing for a BLAS/GPU to chew on.
- CLI takes no batch args (`<dataset> <epochs> <lr> <hidden>`), so it can't even run the same configs as the rest.

## vs the others
Every other variant starts from mini-batches; this one exists to show what that restructuring alone (see [serial-optimized](../serial-optimized/README.md)) buys before any parallelism is added.
