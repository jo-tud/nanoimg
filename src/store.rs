use anyhow::{Context, Result};
use std::fs::{File, OpenOptions};
use std::io::Write;
#[cfg(unix)]
use std::os::unix::fs::FileExt;
use std::path::Path;
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

pub struct VectorStore {
    dims: usize,
    f32_file: File,
    f32_size: u64,
    pub usearch: Index,
    usearch_path: std::path::PathBuf,
}

impl VectorStore {
    pub fn open(data_dir: &Path, dims: usize) -> Result<Self> {
        let f32_path = data_dir.join("vectors_f32.bin");
        let f32_file = OpenOptions::new()
            .create(true)
            .append(true)
            .read(true)
            .open(&f32_path)
            .context("open vectors_f32.bin")?;
        let f32_size = f32_file.metadata()?.len();

        let usearch_path = data_dir.join("vectors.usearch");
        // IP (inner product) with F32 — equivalent to cosine when vectors are L2-normalised.
        // B1 quantisation only supports Hamming/Tanimoto, not cosine.
        let opts = IndexOptions {
            dimensions: dims,
            metric: MetricKind::IP,
            quantization: ScalarKind::F32,
            ..Default::default()
        };
        let usearch = if usearch_path.exists() {
            let idx = Index::new(&opts)?;
            idx.load(usearch_path.to_str().unwrap())?;
            anyhow::ensure!(idx.dimensions() == dims,
                "vectors.usearch has {} dims, model needs {dims} — run with --reindex",
                idx.dimensions());
            idx
        } else {
            Index::new(&opts)?
        };

        Ok(Self { dims, f32_file, f32_size, usearch, usearch_path })
    }

    pub fn dims(&self) -> usize {
        self.dims
    }

    /// Append a vector; returns byte offset before write.
    pub fn append_f32(&mut self, v: &[f32]) -> Result<u64> {
        assert_eq!(v.len(), self.dims, "vector must be {}-dim", self.dims);
        let offset = self.f32_size;
        let bytes = unsafe {
            std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4)
        };
        self.f32_file.write_all(bytes)?;
        self.f32_size += bytes.len() as u64;
        Ok(offset)
    }

    /// Read a vector at `byte_offset` via pread (single syscall).
    pub fn read_f32(&self, byte_offset: u64) -> Result<Vec<f32>> {
        let mut buf = vec![0u8; self.dims * 4];
        self.f32_file.read_exact_at(&mut buf, byte_offset)
            .context("pread vectors_f32.bin")?;
        let v: Vec<f32> = buf.chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        Ok(v)
    }

    /// Pre-allocate capacity in the HNSW index. Must be called before the first `add`.
    pub fn reserve(&self, n: usize) -> Result<()> {
        let current = self.usearch.capacity();
        if n > current {
            self.usearch.reserve(n).context("usearch reserve")?;
        }
        Ok(())
    }

    /// Add `v` under `key`, replacing any vector already stored for that key.
    pub fn upsert(&self, key: u64, v: &[f32]) -> Result<()> {
        if self.usearch.contains(key) {
            self.usearch.remove(key).context("usearch remove")?;
        }
        if self.usearch.size() >= self.usearch.capacity() {
            self.reserve((self.usearch.capacity() * 2).max(64))?;
        }
        self.usearch.add(key, v).context("usearch add")?;
        Ok(())
    }

    /// Remove `key` from the ANN index (no-op if absent).
    pub fn remove(&self, key: u64) -> Result<()> {
        if self.usearch.contains(key) {
            self.usearch.remove(key).context("usearch remove")?;
        }
        Ok(())
    }

    pub fn save(&self) -> Result<()> {
        self.usearch
            .save(self.usearch_path.to_str().unwrap())
            .context("save usearch index")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir()
            .join(format!("nanoimg_store_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn append_and_read_vector() {
        let dir = tmp_dir("append");
        let mut store = VectorStore::open(&dir, 768).unwrap();
        let v: Vec<f32> = (0..768).map(|i| i as f32 * 0.001).collect();
        let offset = store.append_f32(&v).unwrap();
        assert_eq!(offset, 0);
        let got = store.read_f32(0).unwrap();
        assert_eq!(v, got);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn multiple_vectors() {
        let dir = tmp_dir("multi");
        let mut store = VectorStore::open(&dir, 768).unwrap();
        let v1: Vec<f32> = (0..768).map(|i| i as f32).collect();
        let v2: Vec<f32> = (0..768).map(|i| -(i as f32)).collect();
        let o1 = store.append_f32(&v1).unwrap();
        let o2 = store.append_f32(&v2).unwrap();
        assert_eq!(o1, 0);
        assert_eq!(o2, 768 * 4);
        assert_eq!(store.read_f32(o1).unwrap(), v1);
        assert_eq!(store.read_f32(o2).unwrap(), v2);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn out_of_bounds_read() {
        let dir = tmp_dir("oob");
        let store = VectorStore::open(&dir, 768).unwrap();
        assert!(store.read_f32(0).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn upsert_replaces_existing_key() {
        let dir = tmp_dir("upsert");
        let store = VectorStore::open(&dir, 768).unwrap();
        let mut a = vec![0f32; 768];
        a[0] = 1.0;
        let mut b = vec![0f32; 768];
        b[1] = 1.0;
        store.upsert(7, &a).unwrap();
        store.upsert(7, &b).unwrap(); // used to fail: "Duplicate keys not allowed"
        assert_eq!(store.usearch.size(), 1);
        let hits = store.usearch.search(&b, 1).unwrap();
        assert_eq!(hits.keys, vec![7]);
        assert!(hits.distances[0] < 1e-4, "stale vector still indexed");
        store.remove(7).unwrap();
        store.remove(7).unwrap(); // idempotent
        assert!(!store.usearch.contains(7));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn reopen_with_other_dims_fails() {
        let dir = tmp_dir("dims");
        let store = VectorStore::open(&dir, 768).unwrap();
        store.upsert(1, &vec![1.0; 768]).unwrap();
        store.save().unwrap();
        drop(store);
        assert!(VectorStore::open(&dir, 1024).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn reserve_and_save() {
        let dir = tmp_dir("save");
        let store = VectorStore::open(&dir, 768).unwrap();
        store.reserve(100).unwrap();
        store.save().unwrap();
        assert!(dir.join("vectors.usearch").exists());
        std::fs::remove_dir_all(&dir).ok();
    }
}
