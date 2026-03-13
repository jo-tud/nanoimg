use backends::TextEmbedder;
use clap::Parser;
use std::io::{BufRead, IsTerminal, Write};
use std::path::PathBuf;
use std::process::ExitCode;

mod backends;
mod db;
mod index;
#[cfg(feature = "gpu")]
mod gpu;
mod models;
mod onnx;
mod shape;
mod store;
mod tokenizer;
mod viewer;

// search logic lives in index.rs — results stream live during indexing

#[derive(Parser)]
#[command(
    name = "nanoimg",
    about = "Semantic photo search — just point and ask.",
    version
)]
struct Cli {
    /// Directory of images (or - to read paths from stdin)
    dir: Option<PathBuf>,

    /// Search query
    query: Option<String>,

    /// Max results (0 = no limit, score threshold filters)
    #[arg(short = 'n', long, default_value = "0")]
    limit: usize,

    /// Force full reindex
    #[arg(long)]
    reindex: bool,

    /// Suppress progress output
    #[arg(short = 'q', long)]
    quiet: bool,

    /// Skip interactive image viewer
    #[arg(long)]
    no_display: bool,

    /// Score cutoff: auto (default), none, or a threshold (e.g. 0.2)
    #[arg(long, default_value = "auto")]
    cutoff: String,
}

fn main() -> ExitCode {
    // Reset SIGPIPE to default so piping to head/grep/etc works correctly
    unsafe { libc::signal(libc::SIGPIPE, libc::SIG_DFL); }

    match run() {
        Ok(found) => if found { ExitCode::SUCCESS } else { ExitCode::from(1) },
        Err(e) => {
            eprintln!("nanoimg: {e}");
            ExitCode::from(2)
        }
    }
}

/// Returns Ok(true) if results were found (or no query), Ok(false) if query matched nothing.
fn run() -> anyhow::Result<bool> {
    let cli = Cli::parse();
    let data_dir = data_dir()?;
    std::fs::create_dir_all(&data_dir)?;

    // Piped input mode: read image paths from stdin
    let stdin_piped = !std::io::stdin().is_terminal();
    let use_stdin = stdin_piped
        && (cli.dir.is_none() || cli.dir.as_ref().map(|d| d.as_os_str() == "-").unwrap_or(false));

    if use_stdin {
        let paths: Vec<String> = std::io::stdin().lock().lines()
            .filter_map(|l| l.ok())
            .filter(|l| !l.is_empty())
            .collect();
        if paths.is_empty() {
            eprintln!("No paths on stdin.");
            return Ok(false);
        }
        let results: Vec<(f64, String)> = paths.into_iter().map(|p| (0.0, p)).collect();
        if !cli.no_display {
            viewer::run(&results)?;
        }
        return Ok(true);
    }

    if cli.reindex {
        for name in &["index.dat", "vectors_f32.bin", "vectors.usearch",
                      "index.db", "index.db-wal", "index.db-shm", "source_dir"] {
            let p = data_dir.join(name);
            if p.exists() { std::fs::remove_file(&p)?; }
        }
        if cli.dir.is_none() {
            eprintln!("Index cleared.");
            return Ok(true);
        }
    }

    let dir = cli.dir.ok_or_else(|| {
        anyhow::anyhow!("specify a directory of images\n\nUsage: nanoimg <DIR> [QUERY]")
    })?;
    let dir = std::fs::canonicalize(&dir).unwrap_or_else(|_| dir.clone());
    if !dir.is_dir() {
        anyhow::bail!("not a directory: {}", dir.display());
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

    let cutoff = match cli.cutoff.as_str() {
        "auto" => index::CutoffMode::Auto,
        "none" => index::CutoffMode::None,
        s => {
            let v: f64 = s.parse().map_err(|_| {
                anyhow::anyhow!("invalid --cutoff: {} (use auto, none, or a number like 0.2)", s)
            })?;
            index::CutoffMode::Fixed(v)
        }
    };

    let db = db::Database::open(&data_dir)?;
    let store = store::VectorStore::open(&data_dir)?;
    let results = index::run(
        dir, !cli.reindex, query_vec.as_deref(), cli.limit, db, store, &data_dir, cli.quiet,
        &cutoff,
    )?;

    // Print results to stdout
    if !results.is_empty() {
        for (_, path) in &results {
            println!("{}", path);
        }

        // Launch viewer if interactive and not suppressed
        if !cli.no_display && std::io::stdout().is_terminal()
            && std::io::stderr().is_terminal() && std::io::stdin().is_terminal()
        {
            eprint!("View results? [Y/n] ");
            std::io::stderr().flush()?;
            let mut input = String::new();
            std::io::stdin().read_line(&mut input)?;
            if !input.trim().eq_ignore_ascii_case("n") {
                viewer::run(&results)?;
            }
        }
        Ok(true)
    } else if cli.query.is_some() {
        eprintln!("No results.");
        Ok(false)
    } else {
        Ok(true)
    }
}

fn data_dir() -> anyhow::Result<PathBuf> {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    Ok(PathBuf::from(home).join(".nanoimg"))
}
