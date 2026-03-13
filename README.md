# nanoimg

Semantic image search. ~2500 lines of Rust (+1400 optional GPU).

Point at a folder of images, ask a question in plain English.

```
nanoimg ~/photos "sunset over water"
nanoimg ~/photos "person with dog" -n 5
nanoimg --reindex ~/photos "cat"
nanoimg --reindex
```

## Nano spirit

Everything that matters is from scratch:

| Component | ~lines | Replaces |
|---|---|---|
| ONNX runtime (21 ops) | 1100 | onnxruntime, tract |
| GPU backend (11 WGSL shaders) | 1400 | cuDNN, wonnx |
| Protobuf parser | 100 | prost, protobuf |
| BPE tokenizer | 300 | tokenizers + serde_json |
| Flat-file database | 150 | rusqlite |

[usearch](https://github.com/unum-cloud/usearch) handles HNSW.
BLAS handles matmul. GPU backend uses wgpu compute shaders.
Everything else is hand-rolled.

## How it works

Images are embedded with [SigLIP2](https://huggingface.co/google/siglip2-base-patch16-224)
and searched by cosine similarity against your text query. Models download automatically on
first run (~350 MB to `~/.nanoimg/models/`). Results stream live as batches finish indexing.

## Build

Linux x86_64 + OpenBLAS:

```
dnf install openblas-devel    # Fedora/RHEL
apt install libopenblas-dev   # Debian/Ubuntu

cargo build --release                   # CPU only
cargo build --release --features gpu    # CPU + GPU (Vulkan/Metal/DX12 via wgpu)
```

GPU auto-detects at runtime. Falls back to CPU if no GPU found.

## Test

```
cargo test                              # unit tests (db, vector store)
cargo test -- --ignored                 # integration tests (downloads models + images)
cargo test --features gpu -- --ignored  # GPU correctness + benchmarks
```

## Data

Everything lives in `~/.nanoimg/`:

```
~/.nanoimg/
├── models/              # ONNX models + tokenizer (~1.5 GB, downloaded on first run)
│   ├── siglip2_image.onnx
│   ├── siglip2_text.onnx
│   └── tokenizer.json
├── index.dat            # image metadata (paths, hashes, vector offsets)
├── vectors_f32.bin      # raw 768-dim f32 embeddings
└── vectors.usearch      # HNSW approximate nearest-neighbor index
```

```
nanoimg --reindex   # clear the index (keeps models)
rm -rf ~/.nanoimg   # delete everything including models
```

## License

MIT
