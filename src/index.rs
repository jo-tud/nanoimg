use anyhow::Result;
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::io::{IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime};
use walkdir::WalkDir;

use crate::backends::{cosine, Embedder};
use crate::db::Database;
use crate::store::VectorStore;

const SUPPORTED_EXTS: &[&str] = &["jpg", "jpeg", "png", "tiff", "tif", "webp"];
const CHUNK_SIZE: usize = 64;
const ANN_CANDIDATES: usize = 50;
const EMBED_DIM: usize = 768;

/// Score cutoff strategy for filtering search results.
pub enum CutoffMode {
    /// Adaptive threshold via Otsu's method on the score distribution,
    /// with a noise floor derived from embedding dimensionality.
    Auto,
    /// No cutoff — return all ANN candidates.
    None,
    /// Fixed user-specified threshold.
    Fixed(f64),
}

// ── Adaptive cutoff ────────────────────────────────────────────────────────

/// For L2-normalized embeddings in d dimensions, random cosine similarity
/// has σ ≈ 1/√d. Scores below 3σ are indistinguishable from noise.
/// Above that, Otsu's method finds the natural split between relevant
/// and irrelevant clusters by maximizing between-class variance.
fn adaptive_cutoff(scores: &mut Vec<f64>) -> f64 {
    let noise_sigma = 1.0 / (EMBED_DIM as f64).sqrt();
    let noise_floor = 3.0 * noise_sigma; // ~0.108 for 768-dim

    scores.retain(|&s| s >= noise_floor);
    scores.sort_by(|a, b| a.partial_cmp(b).unwrap());

    if scores.len() < 3 {
        return noise_floor;
    }

    // Otsu's method: find threshold maximizing between-class variance
    let n = scores.len();
    let nf = n as f64;
    let total_sum: f64 = scores.iter().sum();
    let mut best_threshold = scores[0];
    let mut best_variance = 0.0f64;
    let mut w0 = 0.0f64;
    let mut sum0 = 0.0f64;

    for i in 0..n - 1 {
        w0 += 1.0;
        sum0 += scores[i];
        let w1 = nf - w0;
        let mean0 = sum0 / w0;
        let mean1 = (total_sum - sum0) / w1;
        let between_var = w0 * w1 * (mean0 - mean1).powi(2);
        if between_var > best_variance {
            best_variance = between_var;
            best_threshold = (scores[i] + scores[i + 1]) / 2.0;
        }
    }

    best_threshold.max(noise_floor)
}

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
    quiet: bool,
    cutoff: &CutoffMode,
) -> Result<Vec<(f64, String)>> {
    // dir is already canonicalized by caller
    let dir_prefix = dir.to_string_lossy().into_owned();
    let live = !quiet && std::io::stderr().is_terminal();
    let color = live && std::env::var_os("NO_COLOR").is_none();

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
        return Ok(query_vec.map(|qv| rank(qv, limit, &dir_prefix, &db, &store, cutoff)).unwrap_or_default());
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
    let start = Instant::now();

    for chunk in paths.chunks(CHUNK_SIZE) {
        let results: Vec<IndexResult> = chunk
            .par_iter()
            .filter_map(|(path, hash)| {
                let meta = std::fs::metadata(path).ok()?;
                let img = image::open(path)
                    .map_err(|e| eprintln!("skip {}: {}", path.display(), e))
                    .ok()?;
                let img = thumbnail(&img, 1024);
                let embed = match embedder.embed(&img) {
                    Ok(v) if v.len() == EMBED_DIM => v,
                    Ok(_) => { eprintln!("skip {}: bad embed dim", path.display()); return None; }
                    Err(e) => { eprintln!("skip {}: {}", path.display(), e); return None; }
                };
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
                prev_lines = print_live(qv, limit, indexed, total, &dir_prefix, &db, &store, color, cutoff);
            } else {
                progress(indexed, total, &start);
            }
        }
    }

    store.save()?;

    if live {
        clear_lines(prev_lines);
        eprint!("\r\x1b[2K");
    }

    if let Some(qv) = query_vec {
        Ok(rank(qv, limit, &dir_prefix, &db, &store, cutoff))
    } else {
        eprintln!("Indexed {} images.", indexed);
        Ok(vec![])
    }
}

// ── Progress bar ────────────────────────────────────────────────────────────

fn progress(indexed: usize, total: usize, start: &Instant) {
    let pct = indexed as f64 / total.max(1) as f64;
    let filled = (pct * 30.0) as usize;
    let empty = 30 - filled;
    let elapsed = start.elapsed().as_secs_f64();
    let rate = if elapsed > 0.0 { indexed as f64 / elapsed } else { 0.0 };
    let eta = if rate > 0.0 { (total - indexed) as f64 / rate } else { 0.0 };
    eprint!("\r  [{}{}] {}/{} {:.1}/s ETA {:.0}s  ",
        "█".repeat(filled), "░".repeat(empty), indexed, total, rate, eta);
    std::io::stderr().flush().ok();
}

// ── Search helpers ──────────────────────────────────────────────────────────

fn rank(
    qv: &[f32], limit: usize, dir_prefix: &str,
    db: &Database, store: &VectorStore, cutoff: &CutoffMode,
) -> Vec<(f64, String)> {
    // Auto mode needs a wider candidate pool for reliable score distribution
    let candidates = match cutoff {
        CutoffMode::Auto => 500,
        _ => if limit == 0 { 200 } else { (limit * 3).clamp(ANN_CANDIDATES, 500) },
    };
    let candidates = candidates.min(store.usearch.size());
    if candidates == 0 { return vec![]; }
    let results = match store.usearch.search(qv, candidates) {
        Ok(r) => r,
        Err(_) => return vec![],
    };

    // Score all candidates (no filtering yet)
    let mut all_scored: Vec<(f64, String)> = results
        .keys
        .iter()
        .filter_map(|&key| {
            let path = db.get_path_by_image_id(key as i64).ok()?;
            if !path.starts_with(dir_prefix) { return None; }
            let offset = db.get_vec_offset(key).ok()?;
            let v = store.read_f32(offset).ok()?;
            let score = cosine(qv, &v) as f64;
            Some((score, path))
        })
        .collect();

    // Apply cutoff
    let threshold = match cutoff {
        CutoffMode::Auto => {
            let mut scores: Vec<f64> = all_scored.iter().map(|(s, _)| *s).collect();
            adaptive_cutoff(&mut scores)
        }
        CutoffMode::None => 0.0,
        CutoffMode::Fixed(t) => *t,
    };
    all_scored.retain(|(s, _)| *s >= threshold);

    all_scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    if limit > 0 { all_scored.truncate(limit); }
    all_scored
}

fn print_live(
    qv: &[f32], limit: usize,
    indexed: usize, total: usize,
    dir_prefix: &str, db: &Database, store: &VectorStore,
    color: bool, cutoff: &CutoffMode,
) -> usize {
    let scored = rank(qv, limit, dir_prefix, db, store, cutoff);
    if color {
        eprintln!("\x1b[2m[{}/{}]\x1b[0m", indexed, total);
    } else {
        eprintln!("[{}/{}]", indexed, total);
    }
    let mut lines = 1;
    if scored.is_empty() {
        if color {
            eprintln!("  \x1b[2m(no matches yet)\x1b[0m");
        } else {
            eprintln!("  (no matches yet)");
        }
        lines += 1;
    } else {
        for (_, path) in &scored {
            eprintln!("  {}", path);
            lines += 1;
        }
    }
    lines
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
