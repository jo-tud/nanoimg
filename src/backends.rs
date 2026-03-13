use anyhow::{Context, Result};
use image::DynamicImage;
use std::path::Path;

use crate::onnx::{self, OnnxModel, Tensor};
use crate::tokenizer::BpeTokenizer;

// ── Traits ────────────────────────────────────────────────────────────────────

pub trait Embedder: Send + Sync {
    fn embed(&self, img: &DynamicImage) -> Result<Vec<f32>>;
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

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
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

impl Embedder for SigLIP2ImageEmbedder {
    fn embed(&self, img: &DynamicImage) -> Result<Vec<f32>> {
        let pixel_values = Self::preprocess(img);
        let input = Tensor::f32(vec![1, 3, 224, 224], pixel_values);

        #[cfg(feature = "gpu")]
        if let Some(ref gpu) = self.gpu {
            let outputs = gpu.lock().unwrap().run(&self.model, vec![("pixel_values", input)])
                .context("gpu run image model")?;
            let pooler = outputs.get("pooler_output")
                .context("missing pooler_output")?;
            let mut v = pooler.as_f32().to_vec();
            l2_normalize(&mut v);
            return Ok(v);
        }

        let outputs = onnx::run(&self.model, vec![("pixel_values", input)])
            .context("run image model")?;
        let pooler = outputs.get("pooler_output")
            .context("missing pooler_output")?;
        let mut v = pooler.as_f32().to_vec();
        l2_normalize(&mut v);
        Ok(v)
    }
}

// ── SigLIP2 text embedder ───────────────────────────────────────────────────

pub struct SigLIP2TextEmbedder {
    model: OnnxModel,
    tokenizer: BpeTokenizer,
    #[cfg(feature = "gpu")]
    gpu: Option<std::sync::Mutex<crate::gpu::GpuExecutor>>,
}

impl SigLIP2TextEmbedder {
    pub fn load(model_path: &Path, tokenizer_path: &Path) -> Result<Self> {
        let model = OnnxModel::load(model_path)
            .context("load siglip2 text model")?;
        let tokenizer = BpeTokenizer::load(tokenizer_path)
            .context("load tokenizer")?;

        #[cfg(feature = "gpu")]
        let gpu = crate::gpu::GpuContext::try_new().map(|ctx| {
            std::sync::Mutex::new(crate::gpu::GpuExecutor::new(ctx))
        });

        Ok(Self {
            model,
            tokenizer,
            #[cfg(feature = "gpu")]
            gpu,
        })
    }
}

impl TextEmbedder for SigLIP2TextEmbedder {
    fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        let ids = self.tokenizer.encode(text, 64);
        let input = Tensor::i64(vec![1, 64], ids);

        #[cfg(feature = "gpu")]
        if let Some(ref gpu) = self.gpu {
            let outputs = gpu.lock().unwrap().run(&self.model, vec![("input_ids", input)])
                .context("gpu run text model")?;
            let pooler = outputs.get("pooler_output")
                .context("missing pooler_output")?;
            let mut v = pooler.as_f32().to_vec();
            l2_normalize(&mut v);
            return Ok(v);
        }

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
            embedder.embed(&img).expect("embed");
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
