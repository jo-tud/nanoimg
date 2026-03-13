use anyhow::Result;
use backends::TextEmbedder;
use clap::Parser;
use std::path::PathBuf;

mod backends;
mod db;
mod index;
mod models;
mod onnx;
mod store;
mod tokenizer;

// search logic lives in index.rs — results stream live during indexing

#[derive(Parser)]
#[command(
    name = "nanoimg",
    about = "Semantic photo search — just point and ask.",
    version
)]
struct Cli {
    /// Directory of images
    dir: Option<PathBuf>,

    /// Search query
    query: Option<String>,

    /// Max results
    #[arg(short = 'n', long, default_value = "10")]
    limit: usize,

    /// Force full reindex
    #[arg(long)]
    reindex: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let data_dir = data_dir()?;
    std::fs::create_dir_all(&data_dir)?;

    if cli.reindex {
        for name in &["index.dat", "vectors_f32.bin", "vectors.usearch",
                      "index.db", "index.db-wal", "index.db-shm", "source_dir"] {
            let p = data_dir.join(name);
            if p.exists() { std::fs::remove_file(&p)?; }
        }
        if cli.dir.is_none() {
            println!("Index cleared.");
            return Ok(());
        }
    }

    let dir = cli.dir.ok_or_else(|| anyhow::anyhow!("specify a directory of images"))?;
    let dir = std::fs::canonicalize(&dir).unwrap_or_else(|_| dir.clone());
    if !dir.is_dir() {
        anyhow::bail!("Not a directory: {}", dir.display());
    }

    models::ensure_ready(&data_dir)?;

    // Embed query up front so results appear as soon as first batch is indexed
    let query_vec = if let Some(ref q) = cli.query {
        let model_dir = data_dir.join("models");
        let embedder = backends::SigLIP2TextEmbedder::load(
            &model_dir.join("siglip2_text.onnx"),
            &model_dir.join("tokenizer.json"),
        )?;
        Some(embedder.embed_text(q)?)
    } else {
        None
    };

    let db = db::Database::open(&data_dir)?;
    let store = store::VectorStore::open(&data_dir)?;
    index::run(dir, !cli.reindex, query_vec.as_deref(), cli.limit, db, store, &data_dir)?;

    Ok(())
}

fn data_dir() -> Result<PathBuf> {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    Ok(PathBuf::from(home).join(".nanoimg"))
}
