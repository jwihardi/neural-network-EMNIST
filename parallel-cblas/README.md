# parallel-cblas

> **Tools:** C++20 + CBLAS (OpenBLAS, internally threaded).

[serial-optimized](../serial-optimized/README.md) with the four hot matmuls (two forward, two weight-gradient) replaced by `cblas_sgemm`. Everything else — biases, activations, softmax, metrics — is untouched serial code.

## Why it beats parallel-omp

Same thread count, radically better per-op code. OpenBLAS GEMM kernels do everything the hand-rolled loops don't:

- **Cache tiling + packed panels** — operands are copied into blocked, contiguous layouts so every level of cache is reused to near its capacity, instead of streaming from DRAM.
- **Register-blocked microkernels** — hand-tuned AVX-512 inner kernels keep ~all FMA units busy; the naive loops reach a fraction of that.
- **The `alpha`/`beta` fold** — the weight update runs as `W = -lr/B · dZ·Xᵀ + 1.0 · W` in a single sgemm, deleting the separate scale-and-subtract pass over the weights entirely.

So where OMP saturated the memory bus with inefficient streams, this actually converts bandwidth into FLOPs: 12.6 s on byclass (3 epochs, hidden 512, batch 128), ~4.4× over OMP with far less CPU burn.

## What still limits it

- **Amdahl.** Only the GEMMs are parallel; softmax, bias adds, and metrics run serial between every sgemm call, and every op is a full pass over the data in and out of memory.
- **Thread-pool overhead** — dispatching OpenBLAS threads costs more than small GEMMs are worth, so tiny batches/hidden sizes lose to plain serial.

## Next rung

[parallel-cuda](../parallel-cuda/README.md) applies the same "call a GEMM library" idea on hardware with ~10× the bandwidth — and then removes the two costs this version can't: data movement between ops, and the serial gaps between them.
