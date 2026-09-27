use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;

/// Tests share one data dir (and run `--reindex`), so they must not run concurrently.
static SERIAL: Mutex<()> = Mutex::new(());

/// Public domain speed images from Wikimedia Commons.
/// All are US government works (NASA / US Navy / USAF).
const IMAGES: &[(&str, &str)] = &[
    (
        "sr71_blackbird.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/9/97/Lockheed_SR-71_Blackbird.jpg/500px-Lockheed_SR-71_Blackbird.jpg",
    ),
    (
        "x15_rocket.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d6/X-15_in_flight.jpg/500px-X-15_in_flight.jpg",
    ),
    (
        "apollo11_launch.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/16/Apollo_11_Launch_-_GPN-2000-000630.jpg/500px-Apollo_11_Launch_-_GPN-2000-000630.jpg",
    ),
    (
        "shuttle_columbia.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Space_Shuttle_Columbia_launching.jpg/500px-Space_Shuttle_Columbia_launching.jpg",
    ),
    (
        "fa18_sonic_boom.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f5/FA-18_Hornet_breaking_sound_barrier_%287_July_1999%29_-_filtered.jpg/500px-FA-18_Hornet_breaking_sound_barrier_%287_July_1999%29_-_filtered.jpg",
    ),
];

/// Stable test home — models are cached across runs (~1.5 GB).
fn test_home() -> PathBuf {
    let dir = std::env::temp_dir().join("nanoimg_test_home");
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Download images once, reuse across test runs.
fn image_dir() -> PathBuf {
    let dir = std::env::temp_dir().join("nanoimg_test_images");
    std::fs::create_dir_all(&dir).unwrap();
    for &(name, url) in IMAGES {
        let dest = dir.join(name);
        if !dest.exists() {
            download(url, &dest);
        }
    }
    dir
}

fn download(url: &str, dest: &Path) {
    let status = Command::new("curl")
        .args(["-sL", "--fail", "-o"])
        .arg(dest)
        .arg(url)
        .status()
        .expect("curl not found");
    assert!(status.success(), "failed to download {url}");
    let size = dest.metadata().expect("download missing").len();
    assert!(size > 1000, "download too small ({size} B): {url}");
}

fn nanoimg(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_nanoimg"))
        .env("HOME", test_home())
        .args(args)
        .output()
        .expect("failed to run nanoimg")
}

#[test]
#[ignore] // network + ~1.5 GB model download on first run
fn index_search_and_reindex() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let dir = image_dir();
    let dir_str = dir.to_str().unwrap();

    // Full reindex + search
    let out = nanoimg(&["--reindex", dir_str, "rocket launch"]);
    assert!(
        out.status.success(),
        "index failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(!stdout.trim().is_empty(), "expected search results");
    // All results should be from our test directory
    for line in stdout.lines() {
        assert!(
            line.starts_with(dir_str),
            "result not from test dir: {line}"
        );
    }

    // Search from saved index (no reindex)
    let out = nanoimg(&[dir_str, "fast jet aircraft"]);
    assert!(
        out.status.success(),
        "search failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(!stdout.trim().is_empty(), "expected results from saved index");

    // index.dat should exist
    let data_dir = test_home().join(".nanoimg");
    assert!(data_dir.join("index.dat").exists());

    // --reindex without dir clears the index
    let out = nanoimg(&["--reindex"]);
    assert!(out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("Index cleared"));
    assert!(!data_dir.join("index.dat").exists());
}

/// Regression: editing an indexed file used to fail every later run with
/// "Duplicate keys not allowed", and exact copies in another folder were never indexed.
#[test]
#[ignore] // network + ~1.5 GB model download on first run
fn edited_and_copied_files() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let src = image_dir();
    let root = std::env::temp_dir().join(format!("nanoimg_test_edit_{}", std::process::id()));
    let (a, b) = (root.join("a"), root.join("b"));
    std::fs::create_dir_all(&a).unwrap();
    std::fs::create_dir_all(&b).unwrap();
    for &(name, _) in &IMAGES[..3] {
        std::fs::copy(src.join(name), a.join(name)).unwrap();
    }
    let (a_str, b_str) = (a.to_str().unwrap(), b.to_str().unwrap());

    let out = nanoimg(&["--reindex", a_str, "rocket"]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));

    // Edit an indexed file (content + size change) → must re-index cleanly
    let edited = a.join(IMAGES[0].0);
    let mut bytes = std::fs::read(&edited).unwrap();
    bytes.extend_from_slice(b"trailing");
    std::fs::write(&edited, bytes).unwrap();
    for _ in 0..2 {
        let out = nanoimg(&[a_str, "aircraft", "--cutoff", "none"]);
        assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert_eq!(stdout.lines().count(), 3, "expected all 3 images: {stdout}");
    }

    // Exact copy in another folder → found there (reuses the stored embedding)
    std::fs::copy(a.join(IMAGES[1].0), b.join("copy.jpg")).unwrap();
    let out = nanoimg(&[b_str, "aircraft", "--cutoff", "none"]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert!(String::from_utf8_lossy(&out.stdout).contains("copy.jpg"));

    std::fs::remove_dir_all(&root).ok();
}
