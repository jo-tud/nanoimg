use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use std::io::{IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

struct FileSpec {
    filename: &'static str,
    url: &'static str,
    /// Accepted SHA-256 digests (several when upstream re-encoded a file)
    sha256: &'static [&'static str],
}

/// A SigLIP2 variant: vision + text encoder sharing one embedding space.
pub struct Model {
    pub name: &'static str,
    pub summary: &'static str,
    pub image_size: usize,
    pub dims: usize,
    /// Images per GPU forward pass. Peak VRAM incl. weights, measured:
    /// base 2.1 GB, large 2.7 GB, so400m 4.5 GB (4.0 GB at batch 4, 5% slower).
    pub gpu_batch: usize,
    /// SigLIP's learned calibration: P(match) = sigmoid(logit_scale · cos + logit_bias).
    /// exp(logit_scale) and logit_bias from google/siglip2-* model.safetensors.
    logit_scale: f64,
    logit_bias: f64,
    image: FileSpec,
    text: FileSpec,
}

macro_rules! hf {
    ($repo:literal, $file:literal) => {
        concat!("https://huggingface.co/onnx-community/", $repo, "/resolve/main/", $file)
    };
}

/// Identical across all SigLIP2 sizes (Gemma tokenizer).
static TOKENIZER: FileSpec = FileSpec {
    filename: "tokenizer.json",
    url: hf!("siglip2-base-patch16-224-ONNX", "tokenizer.json"),
    sha256: &[
        "cb9140fae3ac5122c972d37adf83e1248471a38147ad76f8215c8872c6fd8322", // upstream
        "158ed7e9c7518f251e4c43e934aad5d690e30e5c59168de467166491a6bc5bcf", // compact re-encoding shipped before v0.4
    ],
};

pub static MODELS: &[Model] = &[
    Model {
        name: "base",
        summary: "SigLIP2 B/16 @224, 768-dim — fastest (1.5 GB download)",
        image_size: 224,
        dims: 768,
        gpu_batch: 32,
        logit_scale: 112.67,
        logit_bias: -16.772,
        image: FileSpec {
            filename: "siglip2_image.onnx",
            url: hf!("siglip2-base-patch16-224-ONNX", "onnx/vision_model.onnx"),
            sha256: &["c0573e3f4140c3a7c4e9cc5912bd6b26a033b46a6a8e8af26cbea262b163bcad"],
        },
        text: FileSpec {
            filename: "siglip2_text.onnx",
            url: hf!("siglip2-base-patch16-224-ONNX", "onnx/text_model.onnx"),
            sha256: &["baf12d941beabafafb14f7b4adb38dc15be18681b964a84410ec53d9d65e6293"],
        },
    },
    Model {
        name: "large",
        summary: "SigLIP2 L/16 @256, 1024-dim, fp16 weights — better matches (1.8 GB download)",
        image_size: 256,
        dims: 1024,
        gpu_batch: 16,
        logit_scale: 108.05,
        logit_bias: -16.348,
        image: FileSpec {
            filename: "siglip2-large_image_fp16.onnx",
            url: hf!("siglip2-large-patch16-256-ONNX", "onnx/vision_model_fp16.onnx"),
            sha256: &["f7f719385d406bde52e638069d04fc82453deff9df6aad69426375a78450fd1a"],
        },
        text: FileSpec {
            filename: "siglip2-large_text_fp16.onnx",
            url: hf!("siglip2-large-patch16-256-ONNX", "onnx/text_model_fp16.onnx"),
            sha256: &["bf7195836ddeb5beea9277e689ffa8142cbdbb375e2e4b699487f50234884280"],
        },
    },
    Model {
        name: "so400m",
        summary: "SigLIP2 So400m/16 @384, 1152-dim, fp16 weights — best, slowest (2.3 GB download)",
        image_size: 384,
        dims: 1152,
        gpu_batch: 8,
        logit_scale: 109.86,
        logit_bias: -15.932,
        image: FileSpec {
            filename: "siglip2-so400m_image_fp16.onnx",
            url: hf!("siglip2-so400m-patch16-384-ONNX", "onnx/vision_model_fp16.onnx"),
            sha256: &["7a8059233d8076973f8f754b8bf588674baa3ac4982bd605bd1f25986e85096a"],
        },
        text: FileSpec {
            filename: "siglip2-so400m_text_fp16.onnx",
            url: hf!("siglip2-so400m-patch16-384-ONNX", "onnx/text_model_fp16.onnx"),
            sha256: &["6aed8a7f6fbfbbc1464e57ac7dc3582557bd9da8d7abe761163e2eaf7ad66994"],
        },
    },
];

pub fn find(name: &str) -> Result<&'static Model> {
    MODELS.iter().find(|m| m.name == name).ok_or_else(|| {
        let names: Vec<_> = MODELS.iter().map(|m| m.name).collect();
        anyhow::anyhow!("unknown model {name:?} (choose from: {})", names.join(", "))
    })
}

impl Model {
    pub fn image_model(&self, data_dir: &Path) -> PathBuf {
        data_dir.join("models").join(self.image.filename)
    }

    pub fn text_model(&self, data_dir: &Path) -> PathBuf {
        data_dir.join("models").join(self.text.filename)
    }

    pub fn tokenizer(data_dir: &Path) -> PathBuf {
        data_dir.join("models").join(TOKENIZER.filename)
    }

    /// Cosine similarity at which the model's calibrated match probability is `p`.
    pub fn score_at_probability(&self, p: f64) -> f64 {
        ((p / (1.0 - p)).ln() - self.logit_bias) / self.logit_scale
    }

    /// Where this model's index lives. Embedding spaces differ between models,
    /// so each has its own; `base` keeps the original top-level location.
    pub fn index_dir(&self, data_dir: &Path) -> PathBuf {
        if self.name == "base" { data_dir.to_path_buf() } else { data_dir.join(self.name) }
    }
}

pub fn ensure_ready(data_dir: &Path, model: &Model) -> Result<()> {
    let model_dir = data_dir.join("models");
    let needed = [&model.image, &model.text, &TOKENIZER];
    if needed.iter().all(|s| model_dir.join(s.filename).exists()) {
        return Ok(());
    }
    std::fs::create_dir_all(&model_dir)?;

    // Remove stale files: models from earlier versions, interrupted downloads
    if let Ok(entries) = std::fs::read_dir(&model_dir) {
        let known: std::collections::HashSet<&str> = MODELS.iter()
            .flat_map(|m| [m.image.filename, m.text.filename])
            .chain([TOKENIZER.filename])
            .collect();
        for entry in entries.flatten() {
            if let Some(name) = entry.file_name().to_str() {
                if !known.contains(name) {
                    let _ = std::fs::remove_file(entry.path());
                }
            }
        }
    }

    for spec in needed {
        let dest = model_dir.join(spec.filename);
        if already_present(&dest, spec.sha256)? {
            continue;
        }
        eprintln!("Downloading {}...", spec.filename);
        // Download beside the target and rename once verified, so an interrupted
        // download never leaves a truncated model under the real name
        let part = dest.with_extension("part");
        download_with_progress(spec.url, &part)
            .with_context(|| format!("download {}", spec.filename))?;
        let actual = sha256_file(&part)?;
        if !spec.sha256.contains(&actual.as_str()) {
            std::fs::remove_file(&part)?;
            anyhow::bail!("checksum mismatch for {}: got {actual}", spec.filename);
        }
        std::fs::rename(&part, &dest)?;
    }

    eprintln!("All models ready.");
    Ok(())
}

fn already_present(dest: &Path, sha256: &[&str]) -> Result<bool> {
    if !dest.exists() {
        return Ok(false);
    }
    if sha256.contains(&sha256_file(dest)?.as_str()) {
        Ok(true)
    } else {
        eprintln!(
            "{}: checksum mismatch, re-downloading",
            dest.file_name().unwrap_or_default().to_string_lossy()
        );
        Ok(false)
    }
}

fn download_with_progress(url: &str, dest: &Path) -> Result<()> {
    let resp = ureq::get(url).call().context("HTTP GET")?;
    let total: Option<u64> = resp.headers().get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse().ok());
    // into_reader() is unlimited (read_to_vec() would cap at 10 MB)
    let mut reader = resp.into_body().into_reader();
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

/// Lowercase hex, as stored in index.dat and used for model checksums.
pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
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
    Ok(hex(&hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_hex_is_lowercase_and_stable() {
        // reference: printf nanoimg | sha256sum
        let digest = Sha256::digest(b"nanoimg");
        assert_eq!(hex(&digest), "54d87eed0370effde73433c031cd006bd56666bacc6086b8fd35321fe4f4902d");
        assert_eq!(hex(&[0x00, 0x0f, 0xab]), "000fab");
    }

    #[test]
    fn calibration_inverts_sigmoid() {
        for m in MODELS {
            for p in [1e-4, 3e-4, 0.5, 0.9] {
                let cos = m.score_at_probability(p);
                let back = 1.0 / (1.0 + (-(m.logit_scale * cos + m.logit_bias)).exp());
                assert!((back - p).abs() < 1e-9 * p.max(1e-3), "{} p={p} back={back}", m.name);
            }
        }
        // p = 0.5 sits near cos 0.15 for SigLIP2
        assert!((find("base").unwrap().score_at_probability(0.5) - 0.1489).abs() < 1e-3);
    }
}
