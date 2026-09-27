#!/usr/bin/env python3
"""Evaluate `--cutoff auto` on a labeled Imagenette subset.

    python3 eval/cutoff.py [models...]        # default: base large so400m

Downloads Imagenette (fast.ai, 341 MB) into eval/data/ on first run and builds
150 images (first 15 validation images of each of the 10 classes). Queries in
eval/queries.json name the class that should match, or null when nothing in
the set should. Uses target/release/nanoimg with the models in ~/.nanoimg and a
separate index under eval/data/home, so your own index is untouched.
"""
import json, os, shutil, subprocess, sys, tarfile, urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
BIN = os.path.join(ROOT, "..", "target", "release", "nanoimg")
URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
CLASSES = {"n01440764": "fish", "n02102040": "dog", "n02979186": "cassette", "n03000684": "chainsaw",
           "n03028079": "church", "n03394916": "horn", "n03417042": "garbagetruck", "n03425413": "gaspump",
           "n03445777": "golfball", "n03888257": "parachute"}
PER_CLASS = 15


def prepare():
    images = os.path.join(DATA, "images")
    if os.path.isdir(images):
        return images
    os.makedirs(DATA, exist_ok=True)
    tgz = os.path.join(DATA, "imagenette2-320.tgz")
    if not os.path.exists(tgz):
        print("downloading Imagenette (341 MB)...", flush=True)
        urllib.request.urlretrieve(URL, tgz)
    with tarfile.open(tgz) as t:
        t.extractall(DATA, members=[m for m in t.getmembers() if "/val/" in m.name], filter="data")
    os.makedirs(images)
    for wnid, name in CLASSES.items():
        src = os.path.join(DATA, "imagenette2-320", "val", wnid)
        for i, f in enumerate(sorted(os.listdir(src))[:PER_CLASS]):
            shutil.copy(os.path.join(src, f), os.path.join(images, f"{name}__{i:02d}.jpg"))
    return images


def home():
    """Test HOME whose models/ links to the real ones (no re-download)."""
    h = os.path.join(DATA, "home")
    models = os.path.join(h, ".nanoimg", "models")
    if not os.path.exists(models):
        os.makedirs(os.path.dirname(models), exist_ok=True)
        os.symlink(os.path.expanduser("~/.nanoimg/models"), models)
    return h


def main():
    models = sys.argv[1:] or ["base", "large", "so400m"]
    queries = json.load(open(os.path.join(ROOT, "queries.json")))
    images, env = prepare(), dict(os.environ, HOME=home())
    n_pos = sum(q["cat"] is not None for q in queries)
    for m in models:
        subprocess.run([BIN, "-m", m, "--reindex", images, "-q"], env=env, check=True)
        f1s, precs, recs, empty, false_hits = [], [], [], 0, 0
        for q in queries:
            out = subprocess.run([BIN, "-m", m, images, q["text"], "-q", "--no-display"],
                                 env=env, capture_output=True, text=True).stdout
            got = [os.path.basename(l) for l in out.splitlines()]
            if q["cat"] is None:
                false_hits += len(got)
                continue
            tp = sum(f.startswith(q["cat"] + "__") for f in got)
            p, r = (tp / len(got) if got else 0.0), tp / PER_CLASS
            precs.append(p); recs.append(r); f1s.append(2 * p * r / (p + r) if p + r else 0.0)
            empty += not got
        avg = lambda v: sum(v) / len(v)
        print(f"{m:7s} F1 {avg(f1s):.2f}  precision {avg(precs):.2f}  recall {avg(recs):.2f}  "
              f"empty {empty}/{n_pos}  false hits on {len(queries) - n_pos} no-match queries: {false_hits}")


if __name__ == "__main__":
    main()
