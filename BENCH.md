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
