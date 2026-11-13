"""Table of every evaluated epoch of one run (written by evaluation/run_epochs.py).

Usage: python evaluation/summarize_epochs.py <tag prefix> [--sort fid_ssa]
Prints the table and writes <results>/<prefix>_epochs.csv (results folder: EVAL_RESULTS or evaluation/results).

Columns, SSA-GAN's protocol first:
  fid_ssa     FID, one image per test image (2,933) vs the whole test photos  (SSA-GAN 590: 15.48, paper 15.61)
  is_30k      IS with StackGAN's CUB Inception on 29,330 images, 10 splits    (SSA-GAN 590: 5.10, paper 5.17)
              (only for the epochs run_epochs.py --x10_top evaluated on the 30k set)
  is_3split   the same Inception on the 2,933 images, 3 splits                (SSA-GAN 590: 5.09)
  fid_crop    FID vs the bbox-cropped test images (our main protocol)
  fid_crop_30k, kid, fd_clip, precision, recall, density, coverage, clip_score, R-precision (CLIP, DAMSM)
"""
import argparse, csv, glob, json, os, re
from pathlib import Path

RESULTS = Path(os.environ.get("EVAL_RESULTS", Path(__file__).resolve().parent / "results"))
COLS = [  # (column, source "main" | "x10", json key, index or None, scale, format, lower is better)
    ("fid_ssa", "main", "fid_uncropped", None, 1, "{:.2f}", True),
    ("is_30k", "x10", "inception_score_cub", 0, 1, "{:.2f}", False),
    ("is_3split", "main", "inception_score_cub_3splits", 0, 1, "{:.2f}", False),
    ("fid_crop", "main", "fid", None, 1, "{:.2f}", True),
    ("fid_crop_30k", "x10", "fid", None, 1, "{:.2f}", True),
    ("kid_x1000", "main", "kid", 0, 1000, "{:.2f}", True),
    ("fd_clip_x100", "main", "fd_clip", None, 100, "{:.2f}", True),
    ("precision", "main", "precision", None, 1, "{:.3f}", False),
    ("recall", "main", "recall", None, 1, "{:.3f}", False),
    ("density", "main", "density", None, 1, "{:.3f}", False),
    ("coverage", "main", "coverage", None, 1, "{:.3f}", False),
    ("clip_score", "main", "clip_score", None, 1, "{:.4f}", False),
    ("r_clip", "main", "r_precision_clip", 0, 1, "{:.1f}", False),
    ("r_damsm", "main", "r_precision_damsm", 0, 1, "{:.1f}", False),
]
LOWER = {c[0]: c[6] for c in COLS}


def load_rows(prefix):
    rows = []
    for f in glob.glob(str(RESULTS / f"{prefix}_*.json")):
        m = re.fullmatch(re.escape(prefix) + r"_(\d+)", Path(f).stem)
        if not m:
            continue
        src = {"main": json.load(open(f))}
        x10 = RESULTS / "paper_protocol" / f"{Path(f).stem}_x10.json"
        src["x10"] = json.load(open(x10)) if x10.is_file() else {}
        row = {"epoch": int(m.group(1))}
        for name, where, key, idx, scale, _, _ in COLS:
            v = src[where].get(key)
            row[name] = None if v is None else (v[idx] if idx is not None else v) * scale
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("--sort", default="epoch", help="epoch or a column name, e.g. fid_ssa")
    a = ap.parse_args()
    rows = load_rows(a.prefix)
    if not rows:
        raise SystemExit(f"no {RESULTS}/{a.prefix}_<epoch>.json")
    if a.sort == "epoch":
        rows.sort(key=lambda r: r["epoch"])
    else:   # best first, epochs without the value last
        sign = 1 if LOWER[a.sort] else -1
        rows.sort(key=lambda r: (r[a.sort] is None, sign * (r[a.sort] or 0)))
    best = {}
    for name, *_, low in COLS:
        vals = [r[name] for r in rows if r[name] is not None]
        if vals:
            best[name] = min(vals) if low else max(vals)
    names = ["epoch"] + [c[0] for c in COLS]
    w = max(len(n) for n in names) + 1
    print(" ".join(f"{n:>{w}s}" for n in names))
    for r in rows:
        cells = [f"{r['epoch']:>{w}d}"]
        for name, *_, fmt, _ in COLS:
            v = r[name]
            cells.append(f"{'-' if v is None else fmt.format(v) + ('*' if v == best.get(name) else ''):>{w}s}")
        print(" ".join(cells))
    print("* = best value of the column; fid_ssa / is_30k = SSA-GAN's protocol (see the docstring)")
    out = RESULTS / f"{a.prefix}_epochs.csv"
    with open(out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=names)
        wr.writeheader()
        wr.writerows(sorted(rows, key=lambda r: r["epoch"]))
    print("wrote", out)


if __name__ == "__main__":
    main()
