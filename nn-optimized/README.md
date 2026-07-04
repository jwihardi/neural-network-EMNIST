# nn-optimized

> **Tools:** CUDA 12 + cuBLAS + cuSOLVER (dense Cholesky).

Every other version in this repo makes gradient descent cheaper. This one removes it. Same 784 → hidden → classes network, but the hidden layer is never trained and the output layer has a closed-form answer — training is one pass over the data and one linear solve. No epochs, no learning rate, no schedule.

## The trick

An extreme learning machine / random-features model:

- **W1 and b1 stay at random init.** A random projection followed by ReLU is already a decent feature map — width does the work that backprop would.
- **The output layer is just ridge regression.** With the hidden activations `H` fixed, minimizing squared error over W2 is linear least squares, and the minimizer is exact: `(H Hᵀ + λI) W2ᵀ = H Yᵀ`.

So the whole "training" is: accumulate `H Hᵀ` (hidden × hidden) and `H Yᵀ` (hidden × classes) over the dataset in 16k-sample chunks, add `λ` to the diagonal, Cholesky solve. The chunks stream straight off disk through pinned double buffers — the read of one chunk overlaps the GPU math of the previous, and the full set never sits in GPU memory.

## Numbers

hidden 4096, λ = 1e-5 (total wall time including load):

| dataset | time | test accuracy |
|---|---|---|
| digits | 1.3 s | 97.6% |
| letters | 0.9 s | 84.8% |
| byclass | 2.8 s | 76.7% |

Width is the only knob: byclass at hidden 8192 reaches 79.5% in 7.9 s, at 512 it manages 63%.

## TF32 only where it's safe

The feature GEMMs run in TF32 like the [cuda version](../parallel-cuda/README.md)'s — noise on the random features is just a slightly different random feature map. The Gram accumulation is forced back to fp32: the normal equations square the condition number of `H`, so the same ~1e-3 relative noise inside `H Hᵀ` lands on the small eigenvalues the solve depends on, and accuracy drops to chance.

## The catch

- **Accuracy is bought with width, not iterations.** Cost grows as hidden² × samples in the accumulate and hidden³ in the solve; the return shrinks — byclass plateaus around ~80% where [parallel-cuda](../parallel-cuda/README.md) with momentum reaches 85%.
- Squared error over one-hot targets isn't cross-entropy: the outputs aren't logits, so the reported loss is a softmax over regression scores — comparable in spirit, not in value.
- `H Hᵀ` is hidden² floats — width has a memory ceiling long before the GPU runs out of FLOPs.

Where the rest of the repo climbs a ladder of *making each epoch cheaper*, this trades the ladder for a single algebraic step — and wins whenever "a decent model, now" matters more than the last few points of accuracy.
