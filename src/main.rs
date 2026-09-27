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

    /// Score cutoff: auto (model-calibrated), none, or a cosine threshold (e.g. 0.12)
    #[arg(long, default_value = "auto")]
    cutoff: String,

    /// Print scores alongside paths
    #[arg(short = 's', long)]
    scores: bool,

    /// Embedding model; each keeps its own index
    #[arg(short = 'm', long, env = "NANOIMG_MODEL", default_value = "base", value_parser = model_names())]
    model: String,
}

fn model_names() -> clap::builder::PossibleValuesParser {
    clap::builder::PossibleValuesParser::new(models::MODELS.iter()
        .map(|m| clap::builder::PossibleValue::new(m.name).help(m.summary)))
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
    let model = models::find(&cli.model)?;
    let data_dir = data_dir()?;
    let index_dir = model.index_dir(&data_dir);
    std::fs::create_dir_all(&index_dir)?;

    if cli.reindex {
        for name in &["index.dat", "vectors_f32.bin", "vectors.usearch",
                      "index.db", "index.db-wal", "index.db-shm", "source_dir"] {
            let p = index_dir.join(name);
            if p.exists() { std::fs::remove_file(&p)?; }
        }
        if cli.dir.is_none() {
            eprintln!("Index cleared.");
            return Ok(true);
        }
    }

    // Piped input mode: read image paths from stdin
    let stdin_piped = !std::io::stdin().is_terminal();
    let use_stdin = stdin_piped
        && (cli.dir.is_none() || cli.dir.as_ref().map(|d| d.as_os_str() == "-").unwrap_or(false));

    if use_stdin {
        // Stop at the first read error (lines().filter_map(ok) spun forever on a
        // persistent one); skip lines that aren't UTF-8, as before
        let paths: Vec<String> = std::io::stdin().lock().split(b'\n')
            .map_while(Result::ok)
            .filter_map(|l| String::from_utf8(l).ok())
            .map(|l| l.trim_end_matches('\r').to_string())
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

    let dir = cli.dir.ok_or_else(|| {
        anyhow::anyhow!("specify a directory of images\n\nUsage: nanoimg <DIR> [QUERY]")
    })?;
    let dir = std::fs::canonicalize(&dir).unwrap_or_else(|_| dir.clone());
    if !dir.is_dir() {
        anyhow::bail!("not a directory: {}", dir.display());
    }

    models::ensure_ready(&data_dir, model)?;

    // Embed query up front so results appear as soon as first batch is indexed
    let query_vec = if let Some(ref q) = cli.query {
        let embedder = backends::SigLIP2TextEmbedder::load(
            &model.text_model(&data_dir),
            &models::Model::tokenizer(&data_dir),
        )?;
        Some(embedder.embed_text(q)?)
    } else {
        None
    };

    let min_score = match cli.cutoff.as_str() {
        "auto" => model.score_at_probability(AUTO_MATCH_PROBABILITY),
        "none" => -1.0,
        s => s.parse().map_err(|_| {
            anyhow::anyhow!("invalid --cutoff: {} (use auto, none, or a number like 0.12)", s)
        })?,
    };

    let db = db::Database::open(&index_dir)?;
    let store = store::VectorStore::open(&index_dir, model.dims)?;
    let results = index::run(
        dir, !cli.reindex, query_vec.as_deref(), cli.limit, db, store, &data_dir, model,
        cli.quiet, min_score,
    )?;

    // Print results to stdout
    if !results.is_empty() {
        for (score, path) in &results {
            if cli.scores {
                println!("{:.4}\t{}", score, path);
            } else {
                println!("{}", path);
            }
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
        if cli.cutoff == "none" {
            eprintln!("No results.");
        } else {
            eprintln!("No results above the cutoff (--cutoff none shows the closest matches).");
        }
        Ok(false)
    } else {
        Ok(true)
    }
}

/// `--cutoff auto` keeps images whose SigLIP2 match probability is at least this.
/// Tuned on 150 Imagenette photos × 38 EN/DE queries (see BENCH.md): 86–92%
/// recall at ≥96% precision, 0–2 false hits over 12 queries with no match.
/// The previous Otsu-based cutoff found only 38–44% of matching images.
const AUTO_MATCH_PROBABILITY: f64 = 3e-4;

fn data_dir() -> anyhow::Result<PathBuf> {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    Ok(PathBuf::from(home).join(".nanoimg"))
}
