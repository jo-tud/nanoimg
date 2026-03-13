use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use std::io::{IsTerminal, Read, Write};
use std::path::Path;
use std::time::Instant;

struct ModelSpec {
    filename: &'static str,
    url: &'static str,
    sha256: &'static str,
}

static MODELS: &[ModelSpec] = &[
    ModelSpec {
        filename: "siglip2_image.onnx",
        url: "https://huggingface.co/onnx-community/siglip2-base-patch16-224-ONNX/resolve/main/onnx/vision_model.onnx",
        sha256: "c0573e3f4140c3a7c4e9cc5912bd6b26a033b46a6a8e8af26cbea262b163bcad",
    },
    ModelSpec {
        filename: "siglip2_text.onnx",
        url: "https://huggingface.co/onnx-community/siglip2-base-patch16-224-ONNX/resolve/main/onnx/text_model.onnx",
        sha256: "baf12d941beabafafb14f7b4adb38dc15be18681b964a84410ec53d9d65e6293",
    },
    ModelSpec {
        filename: "tokenizer.json",
        url: "https://huggingface.co/onnx-community/siglip2-base-patch16-224-ONNX/resolve/main/tokenizer.json",
        sha256: "158ed7e9c7518f251e4c43e934aad5d690e30e5c59168de467166491a6bc5bcf",
    },
];

pub fn ensure_ready(data_dir: &Path) -> Result<()> {
    let model_dir = data_dir.join("models");
    if MODELS.iter().all(|s| model_dir.join(s.filename).exists()) {
        return Ok(());
    }
    std::fs::create_dir_all(&model_dir)?;

    // Remove stale model files from earlier versions
    if let Ok(entries) = std::fs::read_dir(&model_dir) {
        let known: std::collections::HashSet<&str> =
            MODELS.iter().map(|s| s.filename).collect();
        for entry in entries.flatten() {
            if let Some(name) = entry.file_name().to_str() {
                if !known.contains(name) {
                    let _ = std::fs::remove_file(entry.path());
                }
            }
        }
    }

    for spec in MODELS {
        let dest = model_dir.join(spec.filename);
        if already_present(&dest, spec.sha256)? {
            continue;
        }
        println!("Downloading {}...", spec.filename);
        download_with_progress(spec.url, &dest)
            .with_context(|| format!("download {}", spec.filename))?;
        verify_sha256(&dest, spec.sha256, spec.filename)?;
    }

    println!("All models ready.");
    Ok(())
}

fn already_present(dest: &Path, sha256: &str) -> Result<bool> {
    if !dest.exists() {
        return Ok(false);
    }
    let unverified = sha256 == "0000000000000000000000000000000000000000000000000000000000000000";
    if unverified {
        return Ok(true);
    }
    let actual = sha256_file(dest)?;
    if actual == sha256 {
        Ok(true)
    } else {
        eprintln!(
            "{}: checksum mismatch, re-downloading",
            dest.file_name().unwrap_or_default().to_string_lossy()
        );
        Ok(false)
    }
}

fn verify_sha256(dest: &Path, sha256: &str, label: &str) -> Result<()> {
    if sha256 == "0000000000000000000000000000000000000000000000000000000000000000" {
        return Ok(());
    }
    let actual = sha256_file(dest)?;
    if actual != sha256 {
        std::fs::remove_file(dest)?;
        anyhow::bail!("checksum mismatch for {label}: expected {sha256} got {actual}");
    }
    Ok(())
}

fn download_with_progress(url: &str, dest: &Path) -> Result<()> {
    let resp = ureq::get(url).call().context("HTTP GET")?;
    let total: Option<u64> = resp.header("content-length").and_then(|v| v.parse().ok());
    let mut reader = resp.into_reader();
    let mut file = std::fs::File::create(dest).context("create dest file")?;
    let mut buf = vec![0u8; 65536];
    let mut downloaded = 0u64;
    let live = std::io::stderr().is_terminal();
    let start = Instant::now();
    loop {
        let n = reader.read(&mut buf).context("read response body")?;
        if n == 0 { break; }
        file.write_all(&buf[..n]).context("write dest file")?;
        downloaded += n as u64;
        if live {
            let elapsed = start.elapsed().as_secs_f64();
            let rate = if elapsed > 0.0 { downloaded as f64 / elapsed } else { 0.0 };
            if let Some(total) = total {
                let pct = downloaded as f64 / total as f64;
                let filled = (pct * 30.0) as usize;
                let empty = 30 - filled.min(30);
                eprint!("\r  [{}{}] {}/{} MB  {:.1} MB/s  ",
                    "█".repeat(filled), "░".repeat(empty),
                    downloaded / 1_000_000, total / 1_000_000,
                    rate / 1_000_000.0);
            } else {
                eprint!("\r  {} MB  {:.1} MB/s  ",
                    downloaded / 1_000_000, rate / 1_000_000.0);
            }
            std::io::stderr().flush().ok();
        }
    }
    if live { eprint!("\r\x1b[2K"); std::io::stderr().flush().ok(); }
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 65536];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 { break; }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}
