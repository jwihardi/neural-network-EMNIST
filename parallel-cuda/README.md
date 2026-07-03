# parallel-cuda

> **Tools:** CUDA 12 + cuBLAS (TF32 tensor cores), custom kernels, CUDA graphs.

Everything lives on the GPU: `Matrix::data` is a device pointer, the matmuls are `cublasSgemm`, the rest is a handful of fused kernels. The host reads back two numbers per epoch.

## Why it beats parallel-cblas

Raw hardware is the obvious part — ~10× the memory bandwidth and TF32 tensor cores for the GEMMs. But a naive port (upload batch, launch 16 ops, download results) squanders that; this network is so small that *overhead*, not math, is the enemy. Three structural changes remove it:

- **The dataset never moves.** Raw `uint8` pixels upload once; a kernel normalizes and stores them transposed (pixel-major), so a batch is a **pointer offset** with a leading-dimension stride straight into cuBLAS. No per-batch copies, no gather kernel, no staging buffer — where cblas re-walks the dataset from host memory every batch.
- **Fused kernels close the serial gaps.** bias+ReLU in-place, bias+softmax, ReLU-backward as an in-place mask, one-hot subtract folded into the metrics kernel. What cblas runs as serial CPU passes between sgemms is here a couple of microsecond-scale kernels — 11 launches per batch instead of 16.
- **CUDA graphs delete launch overhead.** At these sizes each op takes ~µs but costs ~5–10 µs to launch — the GPU idles between launches. Epoch 1 runs eagerly (warming up cuBLAS), then the whole epoch (~5,400 batches on byclass) is stream-captured once and every later epoch is a single `cudaGraphLaunch`.

## Performance

2.7 s on the byclass reference config: ~1 s fixed (CUDA context + reading 547 MB off disk) + ~0.37 s per epoch. ~4.4× over cblas, ~20× over OMP. Per-epoch cost is now bound by GEMM memory traffic — the practical ceiling for this network without changing precision or batch size (batch 1024 is ~30% faster per epoch, essentially free).

## What limits it

- Needs an NVIDIA GPU, and the ~1 s startup dominates tiny runs — cblas wins 1-epoch `digits`-sized jobs.
- TF32 rounds the 4th decimal of loss/accuracy vs the CPU versions.
- The captured graph bakes in the batch sequence — per-epoch shuffling would force recapture.

## The chain, summarized

[unoptimized](../serial-unoptimized/README.md) re-reads weights per sample → [optimized](../serial-optimized/README.md) amortizes them over a batch → [omp](../parallel-omp/README.md) adds cores to those loops → [cblas](../parallel-cblas/README.md) makes each op near-optimal → this version keeps the data in one place and deletes the overhead between ops.
