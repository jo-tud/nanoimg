# Benchmark Log

`cargo test --release -- --ignored bench_image --nocapture`

| Version | Image (min) | Text (min) |
|---|---|---|
| v0 — baseline | 1174.6 ms | 390.2 ms |
| v1 — fast paths, Arc, GEMM reorder, dead tensor cleanup | 768.5 ms | 208.9 ms |
