# Benchmark Log

`cargo test --release -- --ignored bench_image --nocapture`

## CPU

| Version | Image (min) | Text (min) |
|---|---|---|
| v0 — baseline | 1174.6 ms | 390.2 ms |
| v1 — fast paths, Arc, GEMM reorder, dead tensor cleanup | 768.5 ms | 208.9 ms |
| v2 — current | 696.2 ms | 195.9 ms |

## GPU (`--features gpu`)

`cargo test --features gpu --release -- --ignored bench_image_gpu --nocapture`

GPU: NVIDIA GeForce RTX 4070 Laptop GPU

| Version | Image (min) | Text (min) |
|---|---|---|
| v1 — initial (output was NaN) | 424.0 ms | 152.6 ms |
| v2 — pow + matmul batch fix | 422.4 ms | 145.1 ms |
| v3 — simplify (dispatch helper, remove len, collapse ops) | 424.6 ms | 145.9 ms |

## Speedups (v3, min times)

- Image: 696.2 / 424.6 = **1.64x**
- Text: 195.9 / 145.9 = **1.34x**

## End-to-end indexing (v0.4, batched GPU)

`nanoimg --reindex <dir>` on 256 camera JPEGs (230 MB), RTX 4070 Laptop GPU, 22 cores.
Includes decoding; the GPU is otherwise idle.

| Version | Wall time | Images/s |
|---|---|---|
| v0.3 — batch 1, naive 16×16 matmul, Conv on CPU | 107.0 s | 2.4 |
| v0.4 — batch 32, 64×64 register-tiled matmul, Conv/Gemm on GPU, decode overlapped | 26.0 s | 9.8 |

Peak VRAM at batch 32: ~2.1 GB.

## Query latency (v0.4)

Text model now always runs on the CPU, with weights read zero-copy from the mapped file.

| Version | Wall time | Max RSS |
|---|---|---|
| v0.3 — text model copied to RAM and uploaded to GPU | 1.36 s | 2.2 GB |
| v0.4 — CPU, zero-copy mmap | 0.58 s | 0.74 GB |

## Model sizes (v0.4)

256 camera JPEGs, RTX 4070 Laptop GPU. Query = text embedding on CPU + search.

| Model | Index 256 photos | Images/s | Peak VRAM | Query | Query RSS |
|---|---|---|---|---|---|
| `base` (f32) | 26.4 s | 9.7 | 2.1 GB | 0.62 s | 0.74 GB |
| `large` (fp16 → f32) | 86.6 s | 3.0 | 2.7 GB | 1.48 s | 1.84 GB |
| `so400m` (fp16 → f32) | 293.4 s | 0.9 | 4.5 GB | 1.74 s | 2.49 GB |

fp16 weights are widened to f32 on first use; token-embedding rows are widened
per query only (saves ~1 GB RSS for large/so400m).

## Auto cutoff (v0.4)

`python3 eval/cutoff.py` — 150 Imagenette photos (10 classes × 15), 38 queries in
English and German (26 with matches, 12 without, e.g. "Katze", "Strand").
F1/precision/recall averaged over queries with matches.

| Model | Cutoff | F1 | Precision | Recall | False hits (12 no-match queries) |
|---|---|---|---|---|---|
| base | v0.3: 3σ noise floor + Otsu | 0.51 | 0.96 | 0.38 | 0 |
| base | v0.4: P(match) ≥ 3·10⁻⁴ | 0.94 | 0.99 | 0.92 | 2 |
| large | v0.3 | 0.56 | 0.96 | 0.43 | 0 |
| large | v0.4 | 0.90 | 0.99 | 0.86 | 1 |
| so400m | v0.3 | 0.58 | 1.00 | 0.44 | 0 |
| so400m | v0.4 | 0.92 | 0.99 | 0.89 | 0 |

Otsu split the matching images themselves; the calibrated threshold is also
independent of library size. Thresholds from 1·10⁻⁴ to 5·10⁻⁴ perform within ±0.03 F1.
