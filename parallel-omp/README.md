# parallel-omp

> **Tools:** C++20 + OpenMP (`-fopenmp`), all cores.

[serial-optimized](../serial-optimized/README.md) with OpenMP work-sharing. One `#pragma omp parallel` region spans each batch's forward/backward pass, and every matrix op runs an `omp for` inside it — the thread team is created once per batch, not once per op. `nowait` drops barriers between ops that touch independent data, and the metrics loop uses a `reduction(+:correct, tot_loss)`.

## Performance
Scales with cores until memory bandwidth runs out. ~56 s on the byclass reference config (3 epochs, hidden 512, batch 128) — ~4–5× over serial, but burning 14 minutes of CPU time to get there.

## Upsides
- Tiny code delta from serial-optimized; the pragmas are the whole diff.
- No libraries beyond the compiler's OpenMP runtime.

## Downsides
- Hand-rolled GEMM loops don't block/tile for cache the way BLAS does, so throughput per core is far below peak — more threads mostly means hitting the bandwidth wall sooner.
- Implicit barriers between work-shared loops add sync overhead every op, every batch.

## vs the others
Same loops as serial-optimized, just spread across threads. [parallel-cblas](../parallel-cblas/README.md) beats it (~4×) with better single-op code rather than more parallelism; [parallel-cuda](../parallel-cuda/README.md) is ~20× faster.
