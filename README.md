# rq-kmeans

A compact implementation of K-Means clustering and residual quantization (RQ/KMeans-like) utilities.

This repository contains reference implementations and small utilities for running and benchmarking
clustering and vector-quantization experiments. The code is organized for clarity and easy experimentation.

**Key points**
- **Clustering:** Simple K-Means implementation in `kmeans.py` for experimenting with clustering workflows.
- **Quantization kernels:** Helper functions for residual/quantization operations are in `kernels/quantize.py`.
- **Benchmarking:** `benchmark.py` provides a lightweight harness to measure performance and quality.

## Requirements
- Tested on Python 3.12
- See `requirements.txt` for pinned dependencies used by this project.

Setup using a virtual environment and `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Project layout

- `kmeans.py` — K-Means implementation and utilities.
- `benchmark.py` — simple benchmarking harness and example usage.
- `kernels/quantize.py` — quantization kernels and helpers.

## RQKmeans (Residual Quantization)

This project includes utilities for residual quantization (RQ) workflows that are similar in spirit to
RQ/KMeans-based vector quantization. The basic idea implemented in the code is:

- Partition the encoding into stages where each stage fits a small codebook (K-Means) to the current residuals.
- Quantize inputs to nearest centroids from a codebook, subtract the centroid (residual), and repeat for additional stages.
- This layered residual approach reduces quantization error compared to a single-stage codebook at the same total bitrate.

The `kmeans.py` module contains simple nearest-centroid training/assignment utilities that can be composed
into RQ-style multi-stage pipelines for experiments.

## Custom Triton kernels (quantize)

`kernels/quantize.py` provides GPU-accelerated quantization primitives implemented with Triton and CPU fallbacks.
Key exported kernels and utilities:

- `quantize_fwd` — Triton kernel that scans codebooks in blocks and computes nearest-centroid indices for each input
  (optimized for moderate dimensionalities and when the batch is large relative to the codebook).
- `quantize_fwd_mm` — Triton kernel using blocked GEMM-style accumulation across dimensions; designed for larger
  problem shapes (large `D` or larger codebooks) and uses locking/atomic updates across blocks.
- CPU fallbacks — `quantize_cpu_fwd` and `quantize_cpu_fwd_mm` registered for non-GPU execution (use `torch.cdist`).

These Triton kernels use `@triton.autotune` with multiple `triton.Config` entries so Triton can pick good launch
parameters (`BLOCK_B`, `BLOCK_N`, `BLOCK_D`, warps/stages) for the target problem size and GPU.

### TF32 optimization
If running on NVIDIA Ampere (sm_80) GPUs, the code conditionally enables a TF32-style conversion path to accelerate
matrix operations (`FP32_TO_TF32_MAX_PRECISION`), which can provide faster throughput with minimal precision loss for
quantization-distance computations.

## Kernel selection rules

The code provides a high-level helper `kmeans_quantize(x, codebook)` which chooses the best kernel at runtime.
Selection logic (summary):

- If `D` (feature dimensionality) is <= 128 and the batch size `B` is greater than `2 * N` (twice the number of
  codebook entries), the implementation prefers the `quantize_fwd` kernel — this is often faster when the
  dimension is moderate and there are many vectors to assign.
- Otherwise, the `quantize_fwd_mm` path is used (returns `(indices, dists)`), which is tuned for larger `D` or
  situations where a blocked GEMM-style accumulation is more efficient.
- On non-GPU devices or when Triton kernels are not available, CPU kernels (`quantize_cpu_fwd`/`quantize_cpu_fwd_mm`)
  are used as fallbacks (they call `torch.cdist`).

Practical notes:

- The autotuning configurations in `kernels/quantize.py` attempt to pick the best block sizes for given `(B, N, D)`.
- If you add new hardware or change typical problem sizes, re-running experiments or adjusting autotune configs may
  improve performance.
- For deterministic behavior or debugging, prefer CPU kernels or call the specific kernel wrappers directly.

