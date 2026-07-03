# serial-optimized

> **Tools:** plain C++20, one thread, no libraries — same compiler flags as the unoptimized build; the difference is purely code.

Same math as [serial-unoptimized](../serial-unoptimized/README.md), restructured around mini-batches. Activations become `hidden × batch` matrices, so every op is a real GEMM instead of a matrix-vector product.

## Performance
~2.5× faster than the baseline on one core. The matmul loop order (`i, u, b` with the batch dimension innermost) keeps writes contiguous and reuses each loaded weight across the whole batch; `__restrict` pointers let the compiler vectorize the inner loops.

## Upsides
- All the algorithmic structure the parallel versions need, with zero dependencies.
- Weight matrices are read once per batch instead of once per sample — arithmetic intensity scales with batch size.

## Downsides
- Still one core; the same loops OpenBLAS runs at near-peak are hand-rolled here.

## vs the others
This is the shared ancestor: [parallel-omp](../parallel-omp/README.md) adds pragmas to these exact loops, [parallel-cblas](../parallel-cblas/README.md) swaps the loops for `cblas_sgemm`, and [parallel-cuda](../parallel-cuda/README.md) moves the whole thing to the GPU.
