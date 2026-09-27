# nanoimg — notes for contributors and coding agents

Semantic image search (SigLIP2) with a from-scratch ONNX runtime, wgpu GPU backend,
BPE tokenizer, flat-file DB and minifb viewer. Keeping dependencies few is the point:
prefer hand-rolled code over new crates.

## Layout
- `onnx.rs` protobuf parser + CPU executor (22 ops); weights zero-copy from the mmap,
  fp16 widened lazily (Gather reads single rows)
- `gpu.rs` WGSL shaders + executor; `backends.rs` embedders (GPU batching, OOM retry)
- `index.rs` scan/hash/embed/rank; `db.rs` index.dat; `store.rs` vectors + usearch HNSW
- `models.rs` model registry (URLs, sha256, dims, image size, GPU batch, logit scale/bias)
- `tokenizer.rs` Gemma BPE; `viewer.rs` result grid

## Build & test
```
cargo test                                               # unit tests, no models needed
cargo test --release -- --include-ignored --test-threads=1
```
The second needs models in `~/.nanoimg/models` (run `nanoimg <dir>` once). It covers
tokenizer-vs-HF reference, GPU==CPU parity for every installed model, forced GPU errors,
and integration tests (download 5 Wikimedia images). `--test-threads=1` keeps the
model/GPU tests from competing for RAM and VRAM.

CI runs clippy with `-D warnings` on current stable for both feature sets, plus an MSRV
check (1.90). Keep the local toolchain current (`rustup update stable`), or new clippy
lints only show up in CI.

## Conventions
- Compact style (one-line fns, dense numeric loops). **Don't run `cargo fmt`** — the code
  isn't rustfmt-formatted; `needless_range_loop` / `too_many_arguments` are allowed
  in `Cargo.toml` on purpose.
- README line counts: raw lines incl. tests, rounded to 10 (GPU to 100). ONNX runtime =
  `onnx.rs` minus its protobuf section + `shape.rs`. Headline = table sum without GPU.
- Record performance changes in `BENCH.md` (same photo set, idle GPU; other GPU
  processes skew results badly).

## Compatibility constraints
- `index.dat` content hashes are lowercase hex SHA-256 of size + first/last 8 KB;
  changing the scheme forfeits embedding reuse for moved/copied files.
- `base` keeps its index at the top of `~/.nanoimg`; other models use `~/.nanoimg/<name>/`.
  Embedding dims are per model; `VectorStore::open` rejects a mismatch.
- Model files are verified against the sha256 list in `models.rs`; add new hashes rather
  than replacing old ones so existing installs stay valid.

## GPU pitfalls (learned the hard way)
- WGSL arrays indexed by loop variables spill out of registers (naga doesn't unroll):
  use named `vec4` accumulators, as in the matmul shader.
- Dispatches are capped at 65535 groups per dimension: use `GpuExecutor::grid`.
- Buffers are freed only after submission completes, so `FLUSH_EVERY` bounds VRAM.
- After a device error wgpu keeps that pass's VRAM reserved: recreate the device before
  retrying (`GpuRunner::reset`).

## Adding a model / tuning the cutoff
New SigLIP2 variant: add a `Model` entry. Get `logit_scale`/`logit_bias` from the
original `google/siglip2-*` safetensors (exp of the stored scale), and measure peak VRAM
to set `gpu_batch`. Then run `python3 eval/cutoff.py <name>` (150 Imagenette photos ×
38 EN/DE queries) and check `AUTO_MATCH_PROBABILITY` in `main.rs` still holds.

## Releases
Bump `Cargo.toml`, merge to `main`, wait for green CI, then push a lightweight tag
`vX.Y.Z` on that commit. `release.yml` builds and attaches the Linux tarball.
