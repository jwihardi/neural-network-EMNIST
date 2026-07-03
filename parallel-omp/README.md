# parallel-omp

> **Tools:** C++20 + OpenMP (`-fopenmp`), all cores.

[serial-optimized](../serial-optimized/README.md)'s loops, unchanged, spread across threads.

## Why it beats serial-optimized

More cores on already-parallel work. The batched loops are embarrassingly parallel across output rows, so `#pragma omp for` splits them with no algorithm change:

- One `#pragma omp parallel` region spans each batch's whole forward/backward pass — the thread team is created **once per batch**, not once per op, which matters when each op is only microseconds.
- `nowait` removes the implicit barrier between ops that touch independent data; the metrics loop folds into a `reduction(+:correct, tot_loss)`.

## Why the speedup isn't linear

It inherits serial-optimized's weakness instead of fixing it:

- **The per-core code is still naive.** No register blocking, no cache tiling — each thread streams its rows at the same per-core efficiency as the serial build. 16 threads means 16 streams competing for the same memory bus, and DRAM bandwidth saturates long before the cores do.
- **Barriers between dependent ops** (GEMM → bias → activation → GEMM…) sync the whole team many times per batch.

Net: ~4–5× over serial on the byclass reference config (55.7 s), while burning ~14 minutes of aggregate CPU time — poor work-efficiency for the wall time it buys.

## Next rung

[parallel-cblas](../parallel-cblas/README.md) shows the counterintuitive result: better single-op code beats more threads — it's ~4× faster than this while doing *less* parallelism-engineering.
