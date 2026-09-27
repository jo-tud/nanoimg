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
    #[cfg(feature = "gpu")]
    gpu: Option<std::sync::Mutex<crate::gpu::GpuExecutor>>,
}

impl SigLIP2ImageEmbedder {
    pub fn load(model_path: &Path) -> Result<Self> {
        let model = OnnxModel::load(model_path)
            .context("load siglip2 image model")?;

        #[cfg(feature = "gpu")]
        let gpu = crate::gpu::GpuContext::try_new().map(|ctx| {
            eprintln!("GPU: {}", ctx.name);
            std::sync::Mutex::new(crate::gpu::GpuExecutor::new(ctx))
        });
        #[cfg(feature = "gpu")]
        if gpu.is_none() {
            eprintln!("No GPU detected, using CPU");
        }

        Ok(Self {
            model,
            #[cfg(feature = "gpu")]
            gpu,
        })
    }

    fn preprocess(img: &DynamicImage) -> Vec<f32> {
        use image::imageops::FilterType;
        let rgb = img.resize_exact(224, 224, FilterType::Triangle).to_rgb8();
        let mut chw = vec![0f32; 3 * 224 * 224];
        for (x, y, pixel) in rgb.enumerate_pixels() {
            for c in 0..3 {
                chw[c * 224 * 224 + y as usize * 224 + x as usize] =
                    (pixel.0[c] as f32 / 255.0 - 0.5) / 0.5;
            }
        }
        chw
    }
}

/// Images per GPU forward pass. Activations for one image are ~50 MB, so this
/// stays well within a few GB of VRAM while keeping the GPU saturated.
#[cfg(feature = "gpu")]
const GPU_BATCH: usize = 32;

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
        if let Some(ref gpu) = self.gpu {
            let mut out = Vec::with_capacity(imgs.len());
            for batch in imgs.chunks(GPU_BATCH) {
                let pixels: Vec<f32> = batch.par_iter().flat_map_iter(Self::preprocess).collect();
                let input = Tensor::f32(vec![batch.len(), 3, 224, 224], pixels);
                let outputs = gpu.lock().unwrap().run(&self.model, vec![("pixel_values", input)])
                    .context("gpu run image model")?;
                out.extend(pooled_embeddings(&outputs)?);
            }
            return Ok(out);
        }

        // CPU: one image per rayon task keeps every core busy with small GEMMs
        imgs.par_iter().map(|img| {
            let input = Tensor::f32(vec![1, 3, 224, 224], Self::preprocess(img));
            let outputs = onnx::run(&self.model, vec![("pixel_values", input)])
                .context("run image model")?;
            Ok(pooled_embeddings(&outputs)?.remove(0))
        }).collect()
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
        let embedder = SigLIP2ImageEmbedder::load(&dir.join("siglip2_image.onnx"))
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
