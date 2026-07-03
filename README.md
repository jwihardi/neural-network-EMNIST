# neural-network

A 2-layer MLP (784 → hidden → classes) trained on EMNIST, implemented five ways to compare serial vs parallel performance. Same math and same seed everywhere — only the execution strategy changes.

## Dependencies

- CMake 3.16+, a C++20 compiler (g++-14)
- Python 3 (dataset download, benchmark script)
- OpenMP — for `parallel-omp-nn`
- OpenBLAS (or any CBLAS) — for `parallel-cblas-nn`
- CUDA toolkit 12.x + NVIDIA GPU — for `parallel-cuda-nn`

Missing BLAS or CUDA just skips that target with a warning.

## Setup

```sh
python3 import_datasets.py      # one-time ~500 MB EMNIST pull into emnist/
mkdir -p build && cd build && cmake .. && make
```

Binaries land in the repo root.

## Running

```sh
# unoptimized baseline has no batching:
./unoptimized-serial-nn <digits|letters|byclass> <epochs> <lr> <hidden>

# all others:
./<binary> <digits|letters|byclass> <epochs> <lr> <hidden> <train_batch> <eval_batch>

# example:
./parallel-cuda-nn byclass 5 0.05 512 128 100
```

## Benchmark

```sh
./benchmark.py -n 3
```

Sweeps every dataset × hidden size × batch size × model, runs each config `-n` times, and prints comparison tables (avg time ± std, accuracy, geomean speedup). Useful flags: `--datasets`, `--hidden`, `--batches`, `--epochs`, `--skip <model>`, `--timeout`. Timed-out runs show as DNF.

## Implementations

| variant | approach | strengths | weaknesses |
|---|---|---|---|
| `serial-unoptimized` | per-sample SGD, naive loops | simplest to read, the baseline | no batching, extremely slow |
| `serial-optimized` | mini-batches, cache-friendly loop order, `__restrict` | big single-core speedup, no dependencies | still one core |
| `parallel-omp` | optimized-serial + OpenMP across the hot loops | scales with cores, tiny code delta | high CPU burn for its speedup; memory-bandwidth bound |
| `parallel-cblas` | GEMMs handed to OpenBLAS | near-peak CPU throughput, best non-GPU option | only the matmuls are parallel; thread overhead hurts small batches |
| `parallel-cuda` | everything GPU-resident: cuBLAS TF32 GEMMs, fused kernels, dataset stored transposed on device (a batch is a pointer offset), whole epoch captured in a CUDA graph and replayed | fastest by a wide margin; one host sync per epoch | needs an NVIDIA GPU; ~1 s fixed startup (context + data load) dominates tiny runs; TF32 rounds the 4th decimal |

Reference times (byclass 697k images, 3 epochs, hidden 512, batch 128 — RTX 4070 laptop / Zen 4):

| cuda | cblas | omp |
|---|---|---|
| 2.7 s | 12.6 s | 55.7 s |
