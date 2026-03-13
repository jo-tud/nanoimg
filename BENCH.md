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
