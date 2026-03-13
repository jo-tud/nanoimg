# nanoimg

Semantic image search. ~2400 lines of from-scratch Rust (+1400 GPU).

Point at a folder of images, ask a question in plain English.

```
nanoimg ~/photos "sunset over water"
nanoimg ~/photos "person with dog" -n 5
nanoimg --reindex ~/photos "cat"
nanoimg --reindex
```

Results are filtered by an adaptive score cutoff (Otsu's method on the similarity
distribution). Override with `--cutoff none` or `--cutoff 0.2`.

Pipe results to **feh** or any image viewer:

```
nanoimg ~/photos "sunset" --no-display | feh -f -
```

Pipe image paths in:

```
find ~/photos -name '*.jpg' | nanoimg
```

A built-in graphical viewer shows results in a justified grid.
Use `--no-display` or pipe stdout to skip it.

## Nano spirit

Everything that matters is from scratch:

| Component | ~lines | Replaces |
|---|---|---|
| ONNX runtime (21 ops) | 850 | onnxruntime, tract |
| GPU backend (11 WGSL shaders) | 1400 | cuDNN, wonnx |
| Image viewer (minifb) | 680 | feh, eog |
| Protobuf parser | 270 | prost, protobuf |
| BPE tokenizer | 300 | tokenizers + serde_json |
| Flat-file database | 300 | rusqlite |

[usearch](https://github.com/unum-cloud/usearch) handles HNSW.
BLAS handles matmul. GPU backend uses wgpu compute shaders.
Everything else is hand-rolled.

## How it works

Images are embedded with [SigLIP2](https://huggingface.co/google/siglip2-base-patch16-224)
and searched by cosine similarity against your text query. Models download automatically on
first run (~1.5 GB to `~/.nanoimg/models/`). Results stream live as batches finish indexing.

## Build

Linux x86_64 + OpenBLAS:

```
dnf install openblas-devel    # Fedora/RHEL
apt install libopenblas-dev   # Debian/Ubuntu

cargo build --release                          # CPU + GPU (Vulkan/Metal/DX12 via wgpu)
cargo build --release --no-default-features    # CPU only
```

GPU auto-detects at runtime. Falls back to CPU if no GPU found.

## Test

```
cargo test                              # unit tests (db, vector store)
cargo test -- --ignored                 # integration tests (downloads models + images, GPU included by default)
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
