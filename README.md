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
[matrixmultiply](https://github.com/bluss/matrixmultiply) handles matmul (pure Rust, no system BLAS).
GPU backend uses wgpu compute shaders.
Everything else is hand-rolled.

## How it works

Images are embedded with [SigLIP2](https://huggingface.co/blog/siglip2)
and searched by cosine similarity against your text query. Queries work in many languages
("Hund im Schnee" as well as "dog in snow"). Models download automatically on first use
to `~/.nanoimg/models/`. Results stream live as batches finish indexing.

Three model sizes, picked with `--model` (or `NANOIMG_MODEL`):

| Model | Resolution | Dims | Download | Notes |
|---|---|---|---|---|
| `base` (default) | 224 | 768 | 1.5 GB | fastest |
| `large` | 256 | 1024 | 1.8 GB | fp16 weights, noticeably better matches |
| `so400m` | 384 | 1152 | 2.3 GB | fp16 weights, best quality, slowest to index |

```
nanoimg -m large ~/photos "Hochzeit im Garten"
export NANOIMG_MODEL=large    # make it the default
```

Each model keeps its own index, so switching re-indexes once.

## Build

Linux x86_64. No BLAS to install — matmul is pure Rust:

```
cargo build --release                          # CPU + GPU (Vulkan/Metal/DX12 via wgpu)
cargo build --release --no-default-features    # CPU only
```

GPU auto-detects at runtime. Falls back to CPU if no GPU found.

## Test

```
cargo test                              # unit tests (db, vector store)
cargo test -- --ignored                 # tokenizer, GPU and integration tests (downloads models + images)
```

## Data

Everything lives in `~/.nanoimg/`:

```
~/.nanoimg/
├── models/              # ONNX models + tokenizer, downloaded on first use
│   ├── siglip2_image.onnx, siglip2_text.onnx                   # base
│   ├── siglip2-large_{image,text}_fp16.onnx                     # large
│   ├── siglip2-so400m_{image,text}_fp16.onnx                    # so400m
│   └── tokenizer.json                                           # shared
├── index.dat            # base index: image metadata (paths, hashes, vector offsets)
├── vectors_f32.bin      #             raw f32 embeddings
├── vectors.usearch      #             HNSW approximate nearest-neighbor index
├── large/               # same three files for --model large
└── so400m/              # same three files for --model so400m
```

```
nanoimg --reindex   # clear the current model's index (keeps models)
rm -rf ~/.nanoimg   # delete everything including models
```

## License

MIT
