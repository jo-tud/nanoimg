use anyhow::{Context, Result};
use image::DynamicImage;
use rayon::prelude::*;
use std::path::Path;

use crate::onnx::{self, OnnxModel, Tensor};
use crate::tokenizer::BpeTokenizer;

// ── Traits ────────────────────────────────────────────────────────────────────

pub trait Embedder: Send + Sync {
    /// One L2-normalized embedding per image, in order.
    fn embed_batch(&self, imgs: &[DynamicImage]) -> Result<Vec<Vec<f32>>>;
}

pub trait TextEmbedder: Send + Sync {
    fn embed_text(&self, text: &str) -> Result<Vec<f32>>;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

pub fn l2_normalize(v: &mut Vec<f32>) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-6 {
        for x in v.iter_mut() { *x /= norm; }
    }
}

// ── SigLIP2 image embedder ──────────────────────────────────────────────────

pub struct SigLIP2ImageEmbedder {
    model: OnnxModel,
    /// Square input resolution the vision tower was trained at
    size: usize,
    /// None once no GPU was found or it failed even at batch size 1
    #[cfg(feature = "gpu")]
    gpu: std::sync::Mutex<Option<GpuRunner>>,
}

#[cfg(feature = "gpu")]
struct GpuRunner {
    exec: Option<crate::gpu::GpuExecutor>,
    /// Images per forward pass; halved whenever a pass fails (out of VRAM)
    batch: usize,
}

#[cfg(feature = "gpu")]
impl GpuRunner {
    /// Replace the device. wgpu keeps a failed pass's VRAM reserved, so a retry
    /// on the same device fails even at sizes that normally fit.
    fn reset(&mut self) -> Result<()> {
        self.exec = None; // release the old device's memory first
        let ctx = crate::gpu::GpuContext::try_new().context("GPU unavailable")?;
        self.exec = Some(crate::gpu::GpuExecutor::new(ctx));
        Ok(())
    }
}

impl SigLIP2ImageEmbedder {
    pub fn load(model_path: &Path, size: usize, #[allow(unused)] gpu_batch: usize) -> Result<Self> {
        let model = OnnxModel::load(model_path)
            .context("load siglip2 image model")?;

        // NANOIMG_GPU_BATCH overrides the model default (tuning, small GPUs)
        #[cfg(feature = "gpu")]
        let batch = std::env::var("NANOIMG_GPU_BATCH").ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(gpu_batch)
            .max(1);
        #[cfg(feature = "gpu")]
        let gpu = crate::gpu::GpuContext::try_new().map(|ctx| {
            eprintln!("GPU: {}", ctx.name);
            GpuRunner { exec: Some(crate::gpu::GpuExecutor::new(ctx)), batch }
        });
        #[cfg(feature = "gpu")]
        if gpu.is_none() {
            eprintln!("No GPU detected, using CPU");
        }

        Ok(Self {
            model,
            size,
            #[cfg(feature = "gpu")]
            gpu: std::sync::Mutex::new(gpu),
        })
    }

    /// Squash to size×size (no crop, as the SigLIP2 processor does), scale to [-1, 1], CHW.
    fn preprocess(&self, img: &DynamicImage) -> Vec<f32> {
        use image::imageops::FilterType;
        let n = self.size;
        let rgb = img.resize_exact(n as u32, n as u32, FilterType::Triangle).to_rgb8();
        let mut chw = vec![0f32; 3 * n * n];
        for (x, y, pixel) in rgb.enumerate_pixels() {
            for c in 0..3 {
                chw[c * n * n + y as usize * n + x as usize] =
                    (pixel.0[c] as f32 / 255.0 - 0.5) / 0.5;
            }
        }
        chw
    }
}

fn pooled_embeddings(outputs: &std::collections::HashMap<String, Tensor>) -> Result<Vec<Vec<f32>>> {
    let pooler = outputs.get("pooler_output").context("missing pooler_output")?;
    let dim = *pooler.shape.last().context("pooler_output has no dims")?;
    Ok(pooler.as_f32().chunks_exact(dim).map(|v| {
        let mut v = v.to_vec();
        l2_normalize(&mut v);
        v
    }).collect())
}

impl Embedder for SigLIP2ImageEmbedder {
    fn embed_batch(&self, imgs: &[DynamicImage]) -> Result<Vec<Vec<f32>>> {
        #[cfg(feature = "gpu")]
        {
            let mut gpu = self.gpu.lock().unwrap_or_else(|p| p.into_inner());
            if let Some(runner) = gpu.as_mut() {
                match self.embed_gpu(runner, imgs) {
                    Ok(out) => return Ok(out),
                    Err(e) => {
                        eprintln!("{e:#}; continuing on CPU");
                        *gpu = None; // frees the device and its VRAM
                    }
                }
            }
        }

        // CPU: one image per rayon task keeps every core busy with small GEMMs
        imgs.par_iter().map(|img| {
            let input = Tensor::f32(vec![1, 3, self.size, self.size], self.preprocess(img));
            let outputs = onnx::run(&self.model, vec![("pixel_values", input)])
                .context("run image model")?;
            Ok(pooled_embeddings(&outputs)?.remove(0))
        }).collect()
    }
}

#[cfg(feature = "gpu")]
impl SigLIP2ImageEmbedder {
    /// Embed on the GPU, halving the batch size after each failed pass until it
    /// fits. Errors only if a single image fails.
    fn embed_gpu(&self, runner: &mut GpuRunner, imgs: &[DynamicImage]) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(imgs.len());
        let mut done = 0;
        while done < imgs.len() {
            let batch = &imgs[done..(done + runner.batch).min(imgs.len())];
            let pixels: Vec<f32> = batch.par_iter().flat_map_iter(|img| self.preprocess(img)).collect();
            let input = Tensor::f32(vec![batch.len(), 3, self.size, self.size], pixels);
            let exec = runner.exec.as_mut().context("GPU unavailable")?;
            match exec.run(&self.model, vec![("pixel_values", input)]) {
                Ok(outputs) => {
                    out.extend(pooled_embeddings(&outputs)?);
                    done += batch.len();
                }
                Err(e) if runner.batch > 1 => {
                    runner.batch /= 2;
                    eprintln!("{e:#}; retrying with GPU batch size {}", runner.batch);
                    runner.reset()?;
                }
                Err(e) => return Err(e),
            }
        }
        Ok(out)
    }
}

// ── SigLIP2 text embedder ───────────────────────────────────────────────────

/// Runs on the CPU even with the `gpu` feature: a single 64-token query is faster
/// than uploading the 1.1 GB text model to the GPU.
pub struct SigLIP2TextEmbedder {
    model: OnnxModel,
    tokenizer: BpeTokenizer,
}

impl SigLIP2TextEmbedder {
    pub fn load(model_path: &Path, tokenizer_path: &Path) -> Result<Self> {
        let model = OnnxModel::load(model_path)
            .context("load siglip2 text model")?;
        let tokenizer = BpeTokenizer::load(tokenizer_path)
            .context("load tokenizer")?;
        Ok(Self { model, tokenizer })
    }
}

impl TextEmbedder for SigLIP2TextEmbedder {
    fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        // SigLIP2 was trained on lowercased text
        let ids = self.tokenizer.encode(&text.to_lowercase(), 64);
        let input = Tensor::i64(vec![1, 64], ids);
        let outputs = onnx::run(&self.model, vec![("input_ids", input)])
            .context("run text model")?;
        let pooler = outputs.get("pooler_output")
            .context("missing pooler_output")?;
        let mut v = pooler.as_f32().to_vec();
        l2_normalize(&mut v);
        Ok(v)
    }
}

#[cfg(test)]
mod bench {
    use super::*;
    use std::time::Instant;

    fn model_dir() -> std::path::PathBuf {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        std::path::PathBuf::from(home).join(".nanoimg/models")
    }

    #[test]
    #[ignore]
    fn bench_image_embed() {
        let dir = model_dir();
        let embedder = SigLIP2ImageEmbedder::load(&dir.join("siglip2_image.onnx"), 224, 1)
            .expect("load image model");
        let img = DynamicImage::from(image::RgbImage::new(224, 224));
        let n = 10;
        let mut times = Vec::with_capacity(n);
        for _ in 0..n {
            let t = Instant::now();
            embedder.embed_batch(std::slice::from_ref(&img)).expect("embed");
            times.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean: f64 = times.iter().sum::<f64>() / n as f64;
        println!("image embed ({n} iters): min={:.1}ms mean={:.1}ms max={:.1}ms",
            times[0], mean, times[n - 1]);
    }

    #[test]
    #[ignore]
    fn bench_text_embed() {
        let dir = model_dir();
        let embedder = SigLIP2TextEmbedder::load(
            &dir.join("siglip2_text.onnx"),
            &dir.join("tokenizer.json"),
        ).expect("load text model");
        let n = 10;
        let mut times = Vec::with_capacity(n);
        for _ in 0..n {
            let t = Instant::now();
            embedder.embed_text("a photo of a cat").expect("embed_text");
            times.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean: f64 = times.iter().sum::<f64>() / n as f64;
        println!("text embed ({n} iters): min={:.1}ms mean={:.1}ms max={:.1}ms",
            times[0], mean, times[n - 1]);
    }
}
