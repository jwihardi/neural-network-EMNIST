# parallel-cuda

> **Tools:** CUDA 12 + cuBLAS (TF32 tensor cores), custom kernels, CUDA graphs.

Everything lives on the GPU. `Matrix::data` is a device pointer, the five matmuls are `cublasSgemm` (TF32 math mode), and the rest is a handful of fused kernels. The host only reads two numbers (loss, correct) back per epoch.

The wins beyond "use cuBLAS":

- **Dataset resident + transposed** — raw `uint8` pixels upload once, a kernel normalizes and stores them pixel-major, so a batch is a pointer offset with a leading-dimension stride into the full dataset. No per-batch copies, no gather kernel, no `X` buffer.
- **Fused kernels** — bias+ReLU in-place, bias+softmax, ReLU-backward as an in-place mask (`dA1 *= A1 > 0`), and the one-hot subtract folded into the metrics kernel. 11 launches per batch instead of 16.
- **CUDA graph** — epoch 1 runs eagerly (warms up cuBLAS), then the entire epoch (~5400 batches on byclass) is stream-captured once and every later epoch is a single `cudaGraphLaunch`. Launch overhead, previously the dominant cost, disappears.

## Performance
~2.7 s on the byclass reference config (3 epochs, hidden 512, batch 128): ~1 s fixed (context init + reading 547 MB off disk), ~0.37 s per epoch after that. ~4.5× over CBLAS, ~20× over OpenMP. Per-epoch cost is now bound by GEMM memory traffic, not launches — the practical ceiling for this network without changing precision or batch size.

## Upsides
- One host↔device sync per epoch; the GPU never waits on the CPU inside the training loop.
- Batch size scales almost free (batch 1024 is ~30% faster per epoch than 128).

## Downsides
- Needs an NVIDIA GPU; the ~1 s startup dominates small runs, so CBLAS wins 1-epoch `digits`-sized jobs.
- TF32 rounds the 4th decimal of loss/accuracy relative to the CPU versions.
- Graph capture assumes a fixed batch sequence — shuffling between epochs would force recapture.

## vs the others
Same batched structure as [parallel-cblas](../parallel-cblas/README.md), but the data never leaves the device and op launches are amortized to one per epoch. The CPU variants parallelize the work; this one mostly deletes the overhead around it.
