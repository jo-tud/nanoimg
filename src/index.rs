use anyhow::Result;
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::io::{IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::time::SystemTime;
use walkdir::WalkDir;

use crate::backends::{cosine, Embedder};
use crate::db::Database;
use crate::store::VectorStore;

const SUPPORTED_EXTS: &[&str] = &["jpg", "jpeg", "png", "tiff", "tif", "webp"];
const CHUNK_SIZE: usize = 64;
const ANN_CANDIDATES: usize = 50;
const SCORE_THRESHOLD: f64 = 0.05;

struct IndexResult {
    path: PathBuf,
    mtime: i64,
    size: i64,
    embed: Vec<f32>,
    content_hash: String,
}

pub fn run(
    dir: PathBuf,
    update: bool,
    query_vec: Option<&[f32]>,
    limit: usize,
    mut db: Database,
    mut store: VectorStore,
    data_dir: &Path,
) -> Result<()> {
    let dir = std::fs::canonicalize(&dir).unwrap_or(dir);
    let dir_prefix = dir.to_string_lossy().to_string();
    let live = std::io::stderr().is_terminal();

    let paths: Vec<(PathBuf, String)> = WalkDir::new(&dir)
        .follow_links(false)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .filter(|e| {
            e.path()
                .extension()
                .and_then(|s| s.to_str())
                .map(|s| SUPPORTED_EXTS.contains(&s.to_lowercase().as_str()))
                .unwrap_or(false)
        })
        .filter(|e| {
            if !update { return true; }
            let path_str = e.path().to_string_lossy().to_string();
            if let Some((stored_mtime, stored_size)) = db.get_mtime_size(&path_str) {
                if let Ok(meta) = std::fs::metadata(e.path()) {
                    return mtime_secs(&meta) != stored_mtime
                        || meta.len() as i64 != stored_size;
                }
            }
            true
        })
        .filter_map(|e| {
            let hash = quick_hash(e.path())?;
            if db.has_hash(&hash) { return None; }
            Some((e.into_path(), hash))
        })
        .collect();

    // Nothing to index — just search
    if paths.is_empty() {
        if let Some(qv) = query_vec {
            print_final(qv, limit, &dir_prefix, &db, &store);
        }
        return Ok(());
    }

    let img_model = data_dir.join("models").join("siglip2_image.onnx");
    if !img_model.exists() {
        anyhow::bail!("siglip2_image.onnx not found");
    }
    let embedder = crate::backends::SigLIP2ImageEmbedder::load(&img_model)?;

    let total = paths.len();
    store.reserve(store.usearch.size() + total)?;

    let mut indexed = 0usize;
    let mut prev_lines = 0usize;

    for chunk in paths.chunks(CHUNK_SIZE) {
        let results: Vec<IndexResult> = chunk
            .par_iter()
            .filter_map(|(path, hash)| {
                let meta = std::fs::metadata(path).ok()?;
                let img = image::open(path)
                    .map_err(|e| eprintln!("skip {}: {}", path.display(), e))
                    .ok()?;
                let img = thumbnail(&img, 1024);
                let mut embed = embedder.embed(&img).unwrap_or_else(|e| {
                    eprintln!("embed error {}: {}", path.display(), e);
                    vec![0f32; 768]
                });
                if embed.len() != 768 { embed.resize(768, 0.0); }
                Some(IndexResult {
                    path: path.clone(),
                    mtime: mtime_secs(&meta),
                    size: meta.len() as i64,
                    embed,
                    content_hash: hash.clone(),
                })
            })
            .collect();

        db.begin()?;
        for r in &results {
            let path_str = r.path.to_string_lossy().to_string();
            let vec_offset = store.append_f32(&r.embed)?;
            let image_id = db.insert_image(
                &path_str, r.mtime, r.size, vec_offset, &r.content_hash,
            )?;
            store.usearch.add(image_id as u64, &r.embed)?;
        }
        db.commit()?;
        indexed += results.len();

        if live {
            if let Some(qv) = query_vec {
                clear_lines(prev_lines);
                prev_lines = print_live(qv, limit, indexed, total, &dir_prefix, &db, &store);
            } else {
                eprint!("\r\x1b[2K\x1b[2m[{}/{}]\x1b[0m", indexed, total);
                std::io::stderr().flush().ok();
            }
        }
    }

    store.save()?;

    if live {
        clear_lines(prev_lines);
        if query_vec.is_none() {
            eprint!("\r\x1b[2K");
        }
    }

    if let Some(qv) = query_vec {
        print_final(qv, limit, &dir_prefix, &db, &store);
    } else {
        println!("Indexed {} images.", indexed);
    }

    Ok(())
}

// ── Search helpers ──────────────────────────────────────────────────────────

fn rank(
    qv: &[f32], limit: usize, dir_prefix: &str,
    db: &Database, store: &VectorStore,
) -> Vec<(f64, u64)> {
    let results = match store.usearch.search(qv, ANN_CANDIDATES) {
        Ok(r) => r,
        Err(_) => return vec![],
    };
    let mut scored: Vec<(f64, u64)> = results
        .keys
        .iter()
        .filter_map(|&key| {
            // Only include results under the searched directory
            let path = db.get_path_by_image_id(key as i64).ok()?;
            if !path.starts_with(dir_prefix) { return None; }
            let offset = db.get_vec_offset(key).ok()?;
            let v = store.read_f32(offset).ok()?;
            let score = cosine(qv, &v) as f64;
            (score >= SCORE_THRESHOLD).then_some((score, key))
        })
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    scored.truncate(limit);
    scored
}

fn print_live(
    qv: &[f32], limit: usize,
    indexed: usize, total: usize,
    dir_prefix: &str, db: &Database, store: &VectorStore,
) -> usize {
    let scored = rank(qv, limit, dir_prefix, db, store);
    eprintln!("\x1b[2m[{}/{}]\x1b[0m", indexed, total);
    let mut lines = 1;
    if scored.is_empty() {
        eprintln!("  \x1b[2m(no matches yet)\x1b[0m");
        lines += 1;
    } else {
        for &(_, id) in &scored {
            if let Ok(path) = db.get_path_by_image_id(id as i64) {
                eprintln!("  {}", path);
                lines += 1;
            }
        }
    }
    lines
}

fn print_final(
    qv: &[f32], limit: usize, dir_prefix: &str,
    db: &Database, store: &VectorStore,
) {
    let scored = rank(qv, limit, dir_prefix, db, store);
    if scored.is_empty() {
        println!("No results.");
    } else {
        for &(_, id) in &scored {
            if let Ok(path) = db.get_path_by_image_id(id as i64) {
                println!("{}", path);
            }
        }
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn clear_lines(n: usize) {
    for _ in 0..n {
        eprint!("\x1b[A\x1b[2K");
    }
    std::io::stderr().flush().ok();
}

/// Fast content fingerprint: SHA-256 of file size + first 8KB.
fn quick_hash(path: &Path) -> Option<String> {
    let mut file = std::fs::File::open(path).ok()?;
    let size = file.metadata().ok()?.len();
    let mut buf = [0u8; 8192];
    let n = Read::read(&mut file, &mut buf).ok()?;
    let mut hasher = Sha256::new();
    hasher.update(size.to_le_bytes());
    hasher.update(&buf[..n]);
    Some(format!("{:x}", hasher.finalize()))
}

fn mtime_secs(meta: &std::fs::Metadata) -> i64 {
    meta.modified()
        .ok()
        .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn thumbnail(img: &image::DynamicImage, max_dim: u32) -> image::DynamicImage {
    let (w, h) = (img.width(), img.height());
    if w <= max_dim && h <= max_dim {
        return img.clone();
    }
    img.thumbnail(max_dim, max_dim)
}
