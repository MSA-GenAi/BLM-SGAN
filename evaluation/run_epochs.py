"""Evaluate every saved generator (netG_<epoch>.pth) of one training run, to choose the epoch to report.

Stage 1  every epoch in [--from_epoch, --to_epoch] on the 2,933-image protocol (one image per test image):
         FID vs the crops and vs the whole photos (SSA-GAN's FID), KID, FD_CLIP, precision/recall/density/coverage,
         CLIP score, R-precision (CLIP + DAMSM), Inception Scores.
Stage 2  (--x10_top K) the K best epochs by --rank_by (default fid_uncropped = SSA-GAN's FID) again on 29,330
         images (10 captions per test image): the 30k IS of AttnGAN / DM-GAN / DF-GAN / SSA-GAN, plus 30k FID.
Finished epochs are skipped, so the script can be stopped and started again. Ends with summarize_epochs.py.

Usage:
  python evaluation/run_epochs.py --models <folder with netG_*.pth + text_encoder.pth> --prefix blm_sgan
         [--from_epoch 355] [--to_epoch 600] [--x10_top 10] [--delete_images] [--kind blm] [--bs 64]
Paths: EVAL_CACHE (images, features) and EVAL_RESULTS (JSON results) default to evaluation/cache, evaluation/results.
An md5sums.txt in the models folder (lines "<md5>  <file>") is checked first.
"""
import argparse, hashlib, json, os, re, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = Path(os.environ.get("EVAL_CACHE", HERE / "cache"))
RESULTS = Path(os.environ.get("EVAL_RESULTS", HERE / "results"))
PY = sys.executable
STAGE1_KEYS = ["clip_score", "fid_uncropped", "inception_score_cub"]


def has(path: Path, keys) -> bool:
    try:
        r = json.load(open(path))
    except (OSError, ValueError):
        return False
    return all(k in r for k in keys)


def call(args, caps, log=None):
    """Run a script in its own process. The exit code is not trusted: on Windows Python sometimes crashes while
    exiting after the results are written, so callers check the output files instead."""
    env = {**os.environ, "EVAL_CAPS_PER_IMAGE": str(caps), "PYTHONWARNINGS": "ignore", "PYTHONIOENCODING": "utf-8"}
    if log is None:
        subprocess.run([PY, "-u", *args], env=env, cwd=HERE.parent)
    else:
        with open(log, "w") as fh:
            subprocess.run([PY, "-u", *args], env=env, cwd=HERE.parent, stdout=fh, stderr=subprocess.STDOUT)


def tail(path, n=20):
    try:
        return "".join(open(path, errors="replace").readlines()[-n:])
    except OSError:
        return ""


def score(tag, ckpt, kind, caps, delete_images, bs):
    res = (RESULTS / "paper_protocol" if tag.endswith("_x10") else RESULTS) / f"{tag}.json"
    t0 = time.time()
    print(f"===== {tag}  ({Path(ckpt).name}, {2933 * caps} images)  {time.strftime('%H:%M:%S')}", flush=True)
    gen_json, gen_npy = CACHE / f"gen_{tag}.json", CACHE / f"gen_{tag}.npy"
    if not (gen_json.is_file() and gen_npy.is_file()):
        call(["evaluation/evaluate.py", "--model", kind, "--ckpt", str(ckpt), "--tag", tag, "--stage", "gen",
              "--bs", str(bs)], caps)
        if not gen_json.is_file():
            raise SystemExit(f"generation of {tag} failed")
    start = time.time()
    log = CACHE / f"judge_{tag}.log"
    call(["evaluation/evaluate.py", "--tag", tag, "--stage", "judge", "--bs", str(bs)], caps, log)
    if not (res.is_file() and res.stat().st_mtime >= start - 1 and has(res, ["clip_score"])):
        raise SystemExit(f"scoring of {tag} failed:\n{tail(log)}")
    call(["evaluation/inception_score.py", "--tags", tag], caps, CACHE / f"is_{tag}.log")
    if not has(res, ["inception_score_cub"]):
        raise SystemExit(f"Inception Score of {tag} failed:\n{tail(CACHE / f'is_{tag}.log')}")
    if delete_images:   # 577 MB (5.8 GB for 30k); the features and results stay
        for f in (gen_npy, gen_json):
            f.unlink(missing_ok=True)
    r = json.load(open(res))
    print(f"  FID {r['fid']:.2f}  FID(SSA) {r.get('fid_uncropped', float('nan')):.2f}  "
          f"IS {r['inception_score_cub'][0]:.2f}  CLIP-S {r['clip_score']:.4f}  "
          f"R-CLIP {r['r_precision_clip'][0]:.1f}%  P {r['precision']:.3f} R {r['recall']:.3f}  "
          f"({time.time() - t0:.0f}s)", flush=True)


def md5_check(folder: Path):
    f = folder / "md5sums.txt"
    if not f.is_file():
        return
    bad = []
    for line in f.read_text().splitlines():
        if not line.strip():
            continue
        want, name = line.split(None, 1)
        p = folder / name.strip().lstrip("*")
        if not p.is_file():
            bad.append(f"{name}: missing")
            continue
        h = hashlib.md5()
        with open(p, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 24), b""):
                h.update(b)
        if h.hexdigest() != want:
            bad.append(f"{name}: md5 differs (corrupted copy)")
    if bad:
        raise SystemExit("md5sums.txt check failed - download these again:\n  " + "\n  ".join(bad))
    print("md5sums.txt: all files match", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--kind", default="blm", choices=["blm", "ssagan"])
    ap.add_argument("--from_epoch", type=int, default=0)
    ap.add_argument("--to_epoch", type=int, default=10 ** 9)
    ap.add_argument("--x10_top", type=int, default=0, help="stage 2 on the K best epochs (0 = off)")
    ap.add_argument("--rank_by", default="fid_uncropped", help="stage-1 field that picks the stage-2 epochs")
    ap.add_argument("--delete_images", action="store_true", help="remove the generated images after scoring")
    ap.add_argument("--bs", type=int, default=16, help="batch size (the noise sequence does not depend on it)")
    a = ap.parse_args()
    models = Path(a.models)
    CACHE.mkdir(parents=True, exist_ok=True); (RESULTS / "paper_protocol").mkdir(parents=True, exist_ok=True)
    if a.kind == "blm" and not (models / "text_encoder.pth").is_file():
        raise SystemExit(f"missing {models / 'text_encoder.pth'}")
    ckpts = sorted((int(m.group(1)), p) for p in models.glob("netG_*.pth")
                   if (m := re.fullmatch(r"netG_(\d+)\.pth", p.name)) and a.from_epoch <= int(m.group(1)) <= a.to_epoch)
    if not ckpts:
        raise SystemExit(f"no netG_<epoch>.pth between epochs {a.from_epoch} and {a.to_epoch} in {models}")
    print(f"{len(ckpts)} generators: epochs {ckpts[0][0]} ... {ckpts[-1][0]}", flush=True)
    md5_check(models)

    if not has(RESULTS / "real_images.json", ["clip_score"]):
        call(["evaluation/evaluate.py", "--real"], 1, CACHE / "real.log")
        if not has(RESULTS / "real_images.json", ["clip_score"]):
            raise SystemExit("the real-image reference failed:\n" + tail(CACHE / "real.log"))

    for ep, ck in ckpts:                                                   # stage 1
        tag = f"{a.prefix}_{ep}"
        if has(RESULTS / f"{tag}.json", STAGE1_KEYS):
            print(f"done already: {tag}", flush=True)
            continue
        score(tag, ck, a.kind, 1, a.delete_images, a.bs)

    if a.x10_top:                                                          # stage 2
        low = not a.rank_by.startswith(("inception_score", "precision", "recall", "density", "coverage",
                                        "clip_score", "r_precision"))
        val = {}
        for ep, _ in ckpts:
            r = json.load(open(RESULTS / f"{a.prefix}_{ep}.json"))
            v = r[a.rank_by]
            val[ep] = v[0] if isinstance(v, list) else v
        top = sorted(val, key=lambda e: val[e] if low else -val[e])[:a.x10_top]
        print(f"stage 2 (29,330 images) for the {len(top)} best epochs by {a.rank_by}: {sorted(top)}", flush=True)
        for ep in sorted(top):
            tag = f"{a.prefix}_{ep}_x10"
            if has(RESULTS / "paper_protocol" / f"{tag}.json", ["clip_score", "inception_score_cub"]):
                print(f"done already: {tag}", flush=True)
                continue
            score(tag, dict(ckpts)[ep], a.kind, 10, a.delete_images, a.bs)

    call(["evaluation/summarize_epochs.py", a.prefix], 1)
    print("ALL DONE", time.strftime("%H:%M:%S"), flush=True)


if __name__ == "__main__":
    main()
