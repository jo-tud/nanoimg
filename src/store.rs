use anyhow::{Context, Result};
use std::fs::{File, OpenOptions};
use std::io::Write;
#[cfg(unix)]
use std::os::unix::fs::FileExt;
use std::path::Path;
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

const DIMS: usize = 768;

pub struct VectorStore {
    f32_file: File,
    f32_size: u64,
    pub usearch: Index,
    usearch_path: std::path::PathBuf,
}

impl VectorStore {
    pub fn open(data_dir: &Path) -> Result<Self> {
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
            dimensions: DIMS,
            metric: MetricKind::IP,
            quantization: ScalarKind::F32,
            ..Default::default()
        };
        let usearch = if usearch_path.exists() {
            let idx = Index::new(&opts)?;
            idx.load(usearch_path.to_str().unwrap())?;
            idx
        } else {
            Index::new(&opts)?
        };

        Ok(Self { f32_file, f32_size, usearch, usearch_path })
    }

    /// Append a 768-dim f32 vector; returns byte offset before write.
    pub fn append_f32(&mut self, v: &[f32]) -> Result<u64> {
        assert_eq!(v.len(), DIMS, "vector must be 768-dim");
        let offset = self.f32_size;
        let bytes = unsafe {
            std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4)
        };
        self.f32_file.write_all(bytes)?;
        self.f32_size += bytes.len() as u64;
        Ok(offset)
    }

    /// Read a 768-dim vector at `byte_offset` via pread (single syscall).
    pub fn read_f32(&self, byte_offset: u64) -> Result<Vec<f32>> {
        let mut buf = vec![0u8; DIMS * 4];
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
        let mut store = VectorStore::open(&dir).unwrap();
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
        let mut store = VectorStore::open(&dir).unwrap();
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
        let store = VectorStore::open(&dir).unwrap();
        assert!(store.read_f32(0).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn reserve_and_save() {
        let dir = tmp_dir("save");
        let store = VectorStore::open(&dir).unwrap();
        store.reserve(100).unwrap();
        store.save().unwrap();
        assert!(dir.join("vectors.usearch").exists());
        std::fs::remove_dir_all(&dir).ok();
    }
}
