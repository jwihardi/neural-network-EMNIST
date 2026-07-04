# neural-network

A 2-layer MLP (784 → hidden → classes) trained on EMNIST, implemented five ways to compare serial vs parallel performance — same math and same seed, only the execution strategy changes. A sixth variant (`nn-optimized`) keeps the architecture but swaps gradient descent for a closed-form solve.

## Dependencies

- CMake 3.16+, a C++20 compiler (g++-14)
- Python 3 (dataset download, benchmark script)
- OpenMP — for `parallel-omp-nn`
- OpenBLAS (or any CBLAS) — for `parallel-cblas-nn`
- CUDA toolkit 12.x + NVIDIA GPU — for `parallel-cuda-nn` and `nn-optimized-nn`

Missing BLAS or CUDA just skips that target with a warning.

## Setup

```sh
python3 import_datasets.py
mkdir -p build && cd build && cmake .. && make
```

Binaries land in the repo root.

## Running

```sh
./unoptimized-serial-nn <digits|letters|byclass> <epochs> <lr> <hidden>

./<binary> <digits|letters|byclass> <epochs> <lr> <hidden> <train_batch> <eval_batch>

./nn-optimized-nn <digits|letters|byclass> <hidden> <lambda>

./parallel-cuda-nn byclass 5 0.05 512 128 100
./nn-optimized-nn byclass 4096 0.00001
```

## Benchmark

```sh
./benchmark.py -n 3
```

Sweeps every dataset × hidden size × batch size × model, runs each config `-n` times, and prints comparison tables (avg time ± std, accuracy, geomean speedup). Useful flags: `--datasets`, `--hidden`, `--batches`, `--epochs`, `--ridge`, `--skip <model>`. Crashed runs show as DNF.

## Implementations

Each folder has its own README with the code-level details.

| variant | approach | strengths | weaknesses |
|---|---|---|---|
| [`serial-unoptimized`](serial-unoptimized/README.md) | per-sample SGD, naive loops | simplest to read, the baseline | no batching, extremely slow |
| [`serial-optimized`](serial-optimized/README.md) | mini-batches, cache-friendly loop order, `__restrict` | big single-core speedup, no dependencies | still one core |
| [`parallel-omp`](parallel-omp/README.md) | optimized-serial + OpenMP across the hot loops | scales with cores, tiny code delta | high CPU burn for its speedup; memory-bandwidth bound |
| [`parallel-cblas`](parallel-cblas/README.md) | GEMMs handed to OpenBLAS | near-peak CPU throughput, best non-GPU option | only the matmuls are parallel; thread overhead hurts small batches |
| [`parallel-cuda`](parallel-cuda/README.md) | everything GPU-resident: cuBLAS TF32 GEMMs, fused kernels, dataset stored transposed on device (a batch is a pointer offset), whole epoch captured in a CUDA graph and replayed, momentum + OneCycle schedule | fastest training by a wide margin; one epoch now beats what plain SGD needed three for | needs an NVIDIA GPU; ~1 s fixed startup (context + data load) dominates tiny runs; TF32 rounds the 4th decimal |
| [`nn-optimized`](nn-optimized/README.md) | frozen random hidden layer, output weights solved in closed form via ridge regression (cuSOLVER Cholesky) — no epochs at all | one pass over the data; fastest route to a decent model | accuracy is bought with hidden width and plateaus below backprop on the hard sets |

Benchmark results (5 epochs, lr 0.05, hidden 512, batch 128, avg of 3 runs — RTX 4070 laptop / Zen 4; unoptimized has no batching, nn-optimized no epochs or batching):

| dataset | unoptimized-serial | optimized-serial | parallel-omp | parallel-cblas | parallel-cuda | nn-optimized |
|---|---|---|---|---|---|---|
| digits | 455.3 s (96.4%) | 208.5 s (97.9%) | 23.8 s (97.9%) | 5.0 s (97.9%) | 1.4 s (99.0%) | 0.5 s (92.5%) |
| letters | 248.8 s (74.2%) | 120.1 s (85.2%) | 13.4 s (85.2%) | 2.8 s (85.2%) | 0.9 s (90.9%) | 0.4 s (70.7%) |
| byclass | DNF | 812.5 s (82.8%) | 83.1 s (82.8%) | 16.5 s (82.8%) | 3.5 s (85.4%) | 0.6 s (63.0%) |

Geomean speedup over optimized-serial across the full 84-config sweep: omp 8.5×, cblas 27×, cuda 73×, nn-optimized 262×. Serial, omp and cblas produce identical accuracy (same math); cuda runs 1–2 points higher from momentum + OneCycle. nn-optimized trails on accuracy at these hidden sizes — width is its only knob (byclass reaches 76.7% at hidden 4096 in 2.8 s). DNF = didn't finish inside the 900 s cap this sweep ran with.

## Credits

The READMEs, `import_datasets.py`, and `benchmark.py` were written with [Claude](https://claude.com/claude-code).
