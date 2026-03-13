use anyhow::{bail, Result};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

struct Record {
    path: String,
    mtime: i64,
    size: i64,
    vec_offset: u64,
    content_hash: String,
}

pub struct Database {
    records: Vec<Record>,
    path_to_id: HashMap<String, usize>,
    hashes: HashSet<String>,
    db_path: PathBuf,
    dirty: bool,
}

const MAGIC: &[u8; 4] = b"NIDB";
const VERSION: u32 = 1;

impl Database {
    pub fn open(data_dir: &Path) -> Result<Self> {
        let db_path = data_dir.join("index.dat");
        let mut db = Self {
            records: Vec::new(),
            path_to_id: HashMap::new(),
            hashes: HashSet::new(),
            db_path,
            dirty: false,
        };
        if db.db_path.exists() {
            db.load()?;
        }
        Ok(db)
    }

    fn load(&mut self) -> Result<()> {
        let data = fs::read(&self.db_path)?;
        if data.len() < 16 {
            bail!("index.dat too short");
        }
        if &data[0..4] != MAGIC {
            bail!("index.dat: bad magic");
        }
        let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
        if version != VERSION {
            bail!("index.dat: unsupported version {version}");
        }
        let count = u64::from_le_bytes(data[8..16].try_into().unwrap()) as usize;
        self.records.reserve(count);
        let mut pos = 16;

        for _ in 0..count {
            if pos + 4 > data.len() { bail!("truncated record"); }
            let path_len = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
            pos += 4;
            if pos + path_len > data.len() { bail!("truncated path"); }
            let path = String::from_utf8(data[pos..pos + path_len].to_vec())?;
            pos += path_len;

            if pos + 24 > data.len() { bail!("truncated record"); }
            let mtime = i64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
            pos += 8;
            let size = i64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
            pos += 8;
            let vec_offset = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
            pos += 8;

            if pos + 4 > data.len() { bail!("truncated record"); }
            let hash_len = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
            pos += 4;
            if pos + hash_len > data.len() { bail!("truncated hash"); }
            let content_hash = String::from_utf8(data[pos..pos + hash_len].to_vec())?;
            pos += hash_len;

            let idx = self.records.len();
            self.path_to_id.insert(path.clone(), idx);
            if !content_hash.is_empty() {
                self.hashes.insert(content_hash.clone());
            }
            self.records.push(Record { path, mtime, size, vec_offset, content_hash });
        }
        Ok(())
    }

    fn save(&self) -> Result<()> {
        let tmp = self.db_path.with_extension("dat.tmp");
        let mut buf = Vec::with_capacity(self.records.len() * 176 + 16);

        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&VERSION.to_le_bytes());
        buf.extend_from_slice(&(self.records.len() as u64).to_le_bytes());

        for r in &self.records {
            let pb = r.path.as_bytes();
            buf.extend_from_slice(&(pb.len() as u32).to_le_bytes());
            buf.extend_from_slice(pb);
            buf.extend_from_slice(&r.mtime.to_le_bytes());
            buf.extend_from_slice(&r.size.to_le_bytes());
            buf.extend_from_slice(&r.vec_offset.to_le_bytes());
            let hb = r.content_hash.as_bytes();
            buf.extend_from_slice(&(hb.len() as u32).to_le_bytes());
            buf.extend_from_slice(hb);
        }

        let mut f = fs::File::create(&tmp)?;
        f.write_all(&buf)?;
        f.sync_all()?;
        fs::rename(&tmp, &self.db_path)?;
        Ok(())
    }

    pub fn get_mtime_size(&self, path: &str) -> Option<(i64, i64)> {
        let &idx = self.path_to_id.get(path)?;
        let r = &self.records[idx];
        Some((r.mtime, r.size))
    }

    pub fn has_hash(&self, hash: &str) -> bool {
        self.hashes.contains(hash)
    }

    pub fn insert_image(
        &mut self, path: &str, mtime: i64, size: i64, vec_offset: u64, content_hash: &str,
    ) -> Result<i64> {
        self.dirty = true;
        if let Some(&idx) = self.path_to_id.get(path) {
            let r = &mut self.records[idx];
            r.mtime = mtime;
            r.size = size;
            r.vec_offset = vec_offset;
            if !content_hash.is_empty() {
                self.hashes.insert(content_hash.to_string());
            }
            r.content_hash = content_hash.to_string();
            Ok((idx + 1) as i64)
        } else {
            let idx = self.records.len();
            self.path_to_id.insert(path.to_string(), idx);
            if !content_hash.is_empty() {
                self.hashes.insert(content_hash.to_string());
            }
            self.records.push(Record {
                path: path.to_string(), mtime, size, vec_offset,
                content_hash: content_hash.to_string(),
            });
            Ok((idx + 1) as i64)
        }
    }

    pub fn get_vec_offset(&self, image_id: u64) -> Result<u64> {
        let idx = image_id.checked_sub(1)
            .ok_or_else(|| anyhow::anyhow!("invalid id 0"))? as usize;
        let r = self.records.get(idx)
            .ok_or_else(|| anyhow::anyhow!("id {image_id} out of range"))?;
        Ok(r.vec_offset)
    }

    pub fn get_path_by_image_id(&self, image_id: i64) -> Result<String> {
        let idx = (image_id - 1) as usize;
        let r = self.records.get(idx)
            .ok_or_else(|| anyhow::anyhow!("id {image_id} out of range"))?;
        Ok(r.path.clone())
    }

    /// Returns all stored paths that start with `prefix`.
    pub fn paths_with_prefix(&self, prefix: &str) -> Vec<String> {
        self.records.iter().map(|r| r.path.clone())
            .filter(|p| p.starts_with(prefix))
            .collect()
    }

    /// Mark a path as removed (clears its record but preserves indices).
    pub fn remove_path(&mut self, path: &str) {
        if let Some(&idx) = self.path_to_id.get(path) {
            self.records[idx].path.clear();
            self.records[idx].content_hash.clear();
            self.path_to_id.remove(path);
            self.dirty = true;
        }
    }

    pub fn begin(&self) -> Result<()> {
        Ok(())
    }

    pub fn commit(&mut self) -> Result<()> {
        if self.dirty {
            self.save()?;
            self.dirty = false;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir()
            .join(format!("nanoimg_db_{name}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn open_empty() {
        let dir = tmp_dir("empty");
        let db = Database::open(&dir).unwrap();
        assert_eq!(db.get_mtime_size("x"), None);
        assert!(!db.has_hash("x"));
        assert!(db.get_vec_offset(1).is_err());
        assert!(db.get_path_by_image_id(1).is_err());
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn insert_and_lookup() {
        let dir = tmp_dir("insert");
        let mut db = Database::open(&dir).unwrap();
        let id = db.insert_image("/a/b.jpg", 100, 200, 42, "h1").unwrap();
        assert_eq!(id, 1);
        assert_eq!(db.get_mtime_size("/a/b.jpg"), Some((100, 200)));
        assert_eq!(db.get_vec_offset(1).unwrap(), 42);
        assert_eq!(db.get_path_by_image_id(1).unwrap(), "/a/b.jpg");
        assert!(db.has_hash("h1"));
        assert!(!db.has_hash("other"));
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn insert_or_replace_preserves_id() {
        let dir = tmp_dir("replace");
        let mut db = Database::open(&dir).unwrap();
        let id1 = db.insert_image("/a.jpg", 1, 10, 0, "h1").unwrap();
        let id2 = db.insert_image("/a.jpg", 2, 20, 99, "h2").unwrap();
        assert_eq!(id1, id2);
        assert_eq!(db.get_mtime_size("/a.jpg"), Some((2, 20)));
        assert_eq!(db.get_vec_offset(1).unwrap(), 99);
        // old hash is still present (harmless false positive)
        assert!(db.has_hash("h1"));
        assert!(db.has_hash("h2"));
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn round_trip_save_reload() {
        let dir = tmp_dir("round");
        {
            let mut db = Database::open(&dir).unwrap();
            db.insert_image("/x.jpg", 10, 20, 30, "abc").unwrap();
            db.insert_image("/y.jpg", 40, 50, 60, "def").unwrap();
            db.commit().unwrap();
        }
        let db = Database::open(&dir).unwrap();
        assert_eq!(db.get_mtime_size("/x.jpg"), Some((10, 20)));
        assert_eq!(db.get_mtime_size("/y.jpg"), Some((40, 50)));
        assert_eq!(db.get_path_by_image_id(1).unwrap(), "/x.jpg");
        assert_eq!(db.get_path_by_image_id(2).unwrap(), "/y.jpg");
        assert_eq!(db.get_vec_offset(1).unwrap(), 30);
        assert_eq!(db.get_vec_offset(2).unwrap(), 60);
        assert!(db.has_hash("abc"));
        assert!(db.has_hash("def"));
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn commit_skips_when_clean() {
        let dir = tmp_dir("clean");
        let mut db = Database::open(&dir).unwrap();
        db.commit().unwrap();
        assert!(!dir.join("index.dat").exists());
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn many_records() {
        let dir = tmp_dir("many");
        {
            let mut db = Database::open(&dir).unwrap();
            for i in 0..200 {
                let id = db.insert_image(
                    &format!("/img/{i}.jpg"), i, i * 10, i as u64 * 100, &format!("h{i}"),
                ).unwrap();
                assert_eq!(id, (i + 1) as i64);
            }
            db.commit().unwrap();
        }
        let db = Database::open(&dir).unwrap();
        for i in 0..200i64 {
            assert_eq!(db.get_mtime_size(&format!("/img/{i}.jpg")), Some((i, i * 10)));
            assert_eq!(db.get_vec_offset((i + 1) as u64).unwrap(), i as u64 * 100);
            assert!(db.has_hash(&format!("h{i}")));
        }
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn id_zero_is_invalid() {
        let dir = tmp_dir("zero");
        let db = Database::open(&dir).unwrap();
        assert!(db.get_vec_offset(0).is_err());
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn replace_then_round_trip() {
        let dir = tmp_dir("replace_rt");
        {
            let mut db = Database::open(&dir).unwrap();
            db.insert_image("/a.jpg", 1, 2, 3, "old").unwrap();
            db.insert_image("/a.jpg", 9, 8, 7, "new").unwrap();
            db.insert_image("/b.jpg", 5, 6, 7, "other").unwrap();
            db.commit().unwrap();
        }
        let db = Database::open(&dir).unwrap();
        assert_eq!(db.get_mtime_size("/a.jpg"), Some((9, 8)));
        assert_eq!(db.get_vec_offset(1).unwrap(), 7);
        assert_eq!(db.get_path_by_image_id(2).unwrap(), "/b.jpg");
        fs::remove_dir_all(&dir).ok();
    }
}
