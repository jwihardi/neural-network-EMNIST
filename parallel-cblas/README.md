# parallel-cblas

> **Tools:** C++20 + CBLAS (OpenBLAS, internally threaded).

[serial-optimized](../serial-optimized/README.md) with the four hot matmuls (two forward, two weight-gradient — the gradients fold the `-lr/batch` scale and accumulation into `alpha`/`beta`) replaced by `cblas_sgemm` calls. Everything else — bias adds, activations, softmax, metrics — stays the same serial code.

## Performance
Fastest CPU variant: ~12.6 s on the byclass reference config (3 epochs, hidden 512, batch 128), ~4× over OpenMP. OpenBLAS brings blocked, vectorized, multi-threaded GEMM kernels that the hand-rolled loops can't match.

## Upsides
- Near-peak CPU throughput for the ~95% of FLOPs that live in the GEMMs, for a ~10-line diff per call site.
- The `alpha`/`beta` trick removes the separate scale-and-subtract passes entirely.

## Downsides
- Amdahl: only the matmuls are parallel; softmax, biases, and metrics are serial between every sgemm.
- OpenBLAS thread pool overhead dominates at small batch sizes / hidden sizes.

## vs the others
Same structure as parallel-omp but wins on per-op quality, not thread count. [parallel-cuda](../parallel-cuda/README.md) applies the identical "call a GEMM library" idea on the GPU and adds what CBLAS can't: keeping the data resident between ops.
