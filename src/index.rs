use anyhow::Result;
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::io::{IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime};

use crate::backends::Embedder;
use crate::db::Database;
use crate::store::VectorStore;

const SUPPORTED_EXTS: &[&str] = &["jpg", "jpeg", "png", "tiff", "tif", "webp", "avif"];
const CHUNK_SIZE: usize = 64;
const ANN_CANDIDATES: usize = 50;

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
    model: &crate::models::Model,
    quiet: bool,
    min_score: f64,
) -> Result<Vec<(f64, String)>> {
    // dir is already canonicalized by caller
    let dir_prefix = format!("{}/", dir.to_string_lossy());
    let live = !quiet && std::io::stderr().is_terminal();
    let color = live && std::env::var_os("NO_COLOR").is_none();

    // Bring the ANN index in line with the database. Records are committed per
    // chunk but the ANN index is only saved at the end, so an interrupted run
    // (Ctrl-C) leaves records without vectors — re-add them from vectors_f32.bin.
    let (live_entries, removed_ids) = db.entries();
    let mut repaired = 0usize;
    for (id, offset) in live_entries {
        if !store.usearch.contains(id) {
            match store.read_f32(offset) {
                Ok(v) => store.upsert(id, &v)?,
                // Vector lost (e.g. crash before it hit disk): forget the record
                // so the file counts as new and gets re-embedded below.
                Err(_) => {
                    if let Ok(path) = db.get_path_by_image_id(id as i64) {
                        db.remove_path(&path);
                    }
                }
            }
            repaired += 1;
        }
    }
    for id in removed_ids {
        if store.usearch.contains(id) {
            store.remove(id)?;
            repaired += 1;
        }
    }
    let mut store_dirty = repaired > 0;
    if repaired > 0 {
        db.commit()?;
        if live { eprintln!("Repaired {} index entries.", repaired); }
    }

    let mut all_files = Vec::new();
    walk_files(&dir, &mut all_files);

    // New or changed files (by mtime/size), with their content hash
    let changed: Vec<(PathBuf, String)> = all_files.iter()
        .filter(|p| {
            p.extension()
                .and_then(|s| s.to_str())
                .map(|s| SUPPORTED_EXTS.contains(&s.to_lowercase().as_str()))
                .unwrap_or(false)
        })
        .filter(|p| {
            if !update { return true; }
            let path_str = p.to_string_lossy().to_string();
            if let Some((stored_mtime, stored_size)) = db.get_mtime_size(&path_str)
                && let Ok(meta) = std::fs::metadata(p)
            {
                return mtime_secs(&meta) != stored_mtime
                    || meta.len() as i64 != stored_size;
            }
            true
        })
        .filter_map(|p| {
            let hash = quick_hash(p)?;
            Some((p.clone(), hash))
        })
        .collect();

    // Content we have embedded before (copies, moves, touched files) reuses its vector
    let (reuse, mut paths): (Vec<_>, Vec<_>) = changed.into_iter()
        .partition(|(_, hash)| db.vec_offset_for_hash(hash).is_some());

    store.reserve(store.usearch.size() + reuse.len() + paths.len())?;

    if !reuse.is_empty() {
        db.begin()?;
        for (path, hash) in reuse {
            let Ok(meta) = std::fs::metadata(&path) else { continue };
            let offset = db.vec_offset_for_hash(&hash).expect("partitioned on hash");
            let Ok(embed) = store.read_f32(offset) else {
                paths.push((path, hash)); // stored vector unreadable → embed again
                continue;
            };
            let image_id = db.insert_image(
                &path.to_string_lossy(), mtime_secs(&meta), meta.len() as i64, offset, &hash,
            )?;
            store.upsert(image_id as u64, &embed)?;
        }
        db.commit()?;
        store_dirty = true;
    }

    // Remove stale entries (files deleted from disk)
    if update {
        let disk_paths: std::collections::HashSet<String> = all_files.iter()
            .map(|p| p.to_string_lossy().into_owned())
            .collect();
        let mut removed = 0;
        for stored in db.paths_with_prefix(&dir_prefix) {
            if !disk_paths.contains(&stored) {
                if let Some(id) = db.remove_path(&stored) {
                    store.remove(id)?;
                }
                removed += 1;
            }
        }
        if removed > 0 {
            db.commit()?;
            store_dirty = true;
            if live { eprintln!("Removed {} stale entries.", removed); }
        }
    }

    // Nothing to embed — just search
    if paths.is_empty() {
        if store_dirty { store.save()?; }
        return Ok(query_vec.map(|qv| rank(qv, limit, &dir_prefix, &db, &store, min_score)).unwrap_or_default());
    }

    let embedder = crate::backends::SigLIP2ImageEmbedder::load(
        &model.image_model(data_dir), model.image_size, model.gpu_batch,
    )?;

    let total = paths.len();

    let mut indexed = 0usize;
    let mut prev_lines = 0usize;
    let start = Instant::now();

    // Decoding runs on the CPU while the previous chunk is embedded (on the GPU),
    // so neither side waits for the other.
    let decode = |chunk: &'_ [(PathBuf, String)]| -> Vec<(PathBuf, String, std::fs::Metadata, image::DynamicImage)> {
        chunk.par_iter()
            .filter_map(|(path, hash)| {
                let meta = std::fs::metadata(path).ok()?;
                let img = open_oriented(path)
                    .map_err(|e| eprintln!("skip {}: {}", path.display(), e))
                    .ok()?;
                Some((path.clone(), hash.clone(), meta, thumbnail(&img, 1024)))
            })
            .collect()
    };
    let chunks: Vec<&[(PathBuf, String)]> = paths.chunks(CHUNK_SIZE).collect();
    let mut next = chunks.first().map(|c| decode(c)).unwrap_or_default();

    for ci in 0..chunks.len() {
        let (meta, imgs): (Vec<_>, Vec<_>) = std::mem::take(&mut next).into_iter()
            .map(|(path, hash, meta, img)| ((path, hash, meta), img))
            .unzip();
        let (embeds, decoded_next) = rayon::join(
            || embedder.embed_batch(&imgs),
            || chunks.get(ci + 1).map(|c| decode(c)).unwrap_or_default(),
        );
        next = decoded_next;
        let embeds = match embeds {
            Ok(e) => e,
            Err(e) => {
                eprintln!("skip {} images: {:#}", imgs.len(), e);
                continue;
            }
        };
        let results: Vec<IndexResult> = meta.into_iter().zip(embeds)
            .filter(|(_, embed)| embed.len() == model.dims)
            .map(|((path, hash, meta), embed)| IndexResult {
                path,
                mtime: mtime_secs(&meta),
                size: meta.len() as i64,
                embed,
                content_hash: hash,
            })
            .collect();

        db.begin()?;
        for r in &results {
            let path_str = r.path.to_string_lossy().to_string();
            let vec_offset = store.append_f32(&r.embed)?;
            let image_id = db.insert_image(
                &path_str, r.mtime, r.size, vec_offset, &r.content_hash,
            )?;
            store.upsert(image_id as u64, &r.embed)?;
        }
        db.commit()?;
        indexed += results.len();

        if live {
            clear_lines(prev_lines);
            eprint!("\r\x1b[2K");
            progress(indexed, total, &start);
            if let Some(qv) = query_vec {
                eprintln!();
                prev_lines = 1 + print_live(qv, limit, indexed, total, &dir_prefix, &db, &store, color, min_score);
            } else {
                prev_lines = 0;
            }
        }
    }

    store.save()?;

    if live {
        clear_lines(prev_lines);
        eprint!("\r\x1b[2K");
    }

    if let Some(qv) = query_vec {
        Ok(rank(qv, limit, &dir_prefix, &db, &store, min_score))
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

/// Top matches in `dir_prefix` with cosine ≥ `min_score`, best first.
fn rank(
    qv: &[f32], limit: usize, dir_prefix: &str,
    db: &Database, store: &VectorStore, min_score: f64,
) -> Vec<(f64, String)> {
    let candidates = if limit == 0 { 500 } else { (limit * 3).clamp(ANN_CANDIDATES, 500) };
    let candidates = candidates.min(store.usearch.size());
    if candidates == 0 { return vec![]; }
    // Filter inside the HNSW search so other folders can't crowd this one out
    let in_dir = |key: u64| db.path_of(key).is_some_and(|p| p.starts_with(dir_prefix));
    let results = match store.usearch.filtered_search(qv, candidates, in_dir) {
        Ok(r) => r,
        Err(_) => return vec![],
    };

    // IP distance is 1 - dot; vectors are L2-normalized, so dot = cosine
    let mut all_scored: Vec<(f64, String)> = results.keys.iter().zip(&results.distances)
        .filter_map(|(&key, &dist)| Some((1.0 - dist as f64, db.path_of(key)?.to_string())))
        .collect();

    all_scored.retain(|(s, _)| *s >= min_score);
    all_scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    if limit > 0 { all_scored.truncate(limit); }
    all_scored
}

fn print_live(
    qv: &[f32], limit: usize,
    indexed: usize, total: usize,
    dir_prefix: &str, db: &Database, store: &VectorStore,
    color: bool, min_score: f64,
) -> usize {
    let scored = rank(qv, limit, dir_prefix, db, store, min_score);
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

/// Fast content fingerprint: SHA-256 of file size + first and last 8KB.
fn quick_hash(path: &Path) -> Option<String> {
    use std::io::{Seek, SeekFrom};
    const BLOCK: u64 = 8192;
    let mut file = std::fs::File::open(path).ok()?;
    let size = file.metadata().ok()?.len();
    let mut buf = [0u8; BLOCK as usize];
    let mut hasher = Sha256::new();
    hasher.update(size.to_le_bytes());
    let n = file.read(&mut buf).ok()?;
    hasher.update(&buf[..n]);
    if size > BLOCK {
        file.seek(SeekFrom::Start(size.saturating_sub(BLOCK).max(BLOCK))).ok()?;
        let n = file.read(&mut buf).ok()?;
        hasher.update(&buf[..n]);
    }
    Some(crate::models::hex(&hasher.finalize()))
}

/// Decode an image and apply its EXIF orientation (portrait phone photos are
/// stored sideways with a rotation tag).
pub fn open_oriented(path: &Path) -> image::ImageResult<image::DynamicImage> {
    use image::ImageDecoder;
    let mut decoder = image::ImageReader::open(path)?.with_guessed_format()?.into_decoder()?;
    let orientation = decoder.orientation()?;
    let mut img = image::DynamicImage::from_decoder(decoder)?;
    img.apply_orientation(orientation);
    Ok(img)
}

/// Image dimensions as displayed, i.e. after EXIF orientation (header only, no decode).
pub fn oriented_dimensions(path: &Path) -> image::ImageResult<(u32, u32)> {
    use image::{metadata::Orientation, ImageDecoder};
    let mut decoder = image::ImageReader::open(path)?.with_guessed_format()?.into_decoder()?;
    let (w, h) = decoder.dimensions();
    Ok(match decoder.orientation()? {
        Orientation::Rotate90 | Orientation::Rotate270
        | Orientation::Rotate90FlipH | Orientation::Rotate270FlipH => (h, w),
        _ => (w, h),
    })
}

fn walk_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let ft = match entry.file_type() {
            Ok(ft) => ft,
            Err(_) => continue,
        };
        if ft.is_dir() {
            walk_files(&entry.path(), out);
        } else if ft.is_file() {
            out.push(entry.path());
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(dims: &[(usize, f32)]) -> Vec<f32> {
        let mut v = vec![0f32; 768];
        for &(i, x) in dims { v[i] = x; }
        crate::backends::l2_normalize(&mut v);
        v
    }

    /// `base` plus small deterministic noise in every dimension, normalized.
    fn noisy(base: &[f32], seed: u64) -> Vec<f32> {
        let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let mut v: Vec<f32> = base.iter().map(|&x| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            x + ((state >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.01
        }).collect();
        crate::backends::l2_normalize(&mut v);
        v
    }

    /// Other folders must not crowd a folder's images out of the ANN candidate pool.
    #[test]
    fn rank_finds_folder_despite_closer_matches_elsewhere() {
        let dir = std::env::temp_dir().join(format!("nanoimg_rank_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let mut db = Database::open(&dir).unwrap();
        let mut store = VectorStore::open(&dir, 768).unwrap();
        store.reserve(700).unwrap();
        let mut add = |path: &str, v: &[f32]| {
            let off = store.append_f32(v).unwrap();
            let id = db.insert_image(path, 0, 0, off, path).unwrap();
            store.upsert(id as u64, v).unwrap();
        };
        // 600 near-perfect matches in /a, one weaker match in /b
        let near = unit(&[(0, 1.0)]);
        for i in 0..600 {
            add(&format!("/a/{i}.jpg"), &noisy(&near, i));
        }
        add("/b/only.jpg", &unit(&[(0, 1.0), (1, 1.0)]));

        let q = unit(&[(0, 1.0)]);
        let hits = rank(&q, 0, "/b/", &db, &store, -1.0);
        assert_eq!(hits.len(), 1, "expected /b/only.jpg, got {hits:?}");
        assert_eq!(hits[0].1, "/b/only.jpg");
        assert!((hits[0].0 - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-3, "score {}", hits[0].0);
        std::fs::remove_dir_all(&dir).ok();
    }
}
