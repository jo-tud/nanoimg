use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::path::Path;

struct ModelSpec {
    filename: &'static str,
    url: &'static str,
    sha256: &'static str,
}

static MODELS: &[ModelSpec] = &[
    ModelSpec {
        filename: "siglip2_image.onnx",
        url: "https://huggingface.co/onnx-community/siglip2-base-patch16-224-ONNX/resolve/main/onnx/vision_model.onnx",
        sha256: "0000000000000000000000000000000000000000000000000000000000000000",
    },
    ModelSpec {
        filename: "siglip2_text.onnx",
        url: "https://huggingface.co/onnx-community/siglip2-base-patch16-224-ONNX/resolve/main/onnx/text_model.onnx",
        sha256: "0000000000000000000000000000000000000000000000000000000000000000",
    },
    ModelSpec {
        filename: "tokenizer.json",
        url: "https://huggingface.co/onnx-community/siglip2-base-patch16-224-ONNX/resolve/main/tokenizer.json",
        sha256: "0000000000000000000000000000000000000000000000000000000000000000",
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
    let pb = match total {
        Some(n) => {
            let bar = ProgressBar::new(n);
            bar.set_style(
                ProgressStyle::with_template(
                    "  {bar:40.cyan/blue} {bytes}/{total_bytes} {bytes_per_sec} eta {eta}",
                )
                .unwrap(),
            );
            bar
        }
        None => {
            let bar = ProgressBar::new_spinner();
            bar.set_style(
                ProgressStyle::with_template("  {spinner} {bytes} downloaded").unwrap(),
            );
            bar
        }
    };
    let mut reader = resp.into_reader();
    let mut file = std::fs::File::create(dest).context("create dest file")?;
    let mut buf = vec![0u8; 65536];
    loop {
        let n = reader.read(&mut buf).context("read response body")?;
        if n == 0 { break; }
        file.write_all(&buf[..n]).context("write dest file")?;
        pb.inc(n as u64);
    }
    pb.finish_and_clear();
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
