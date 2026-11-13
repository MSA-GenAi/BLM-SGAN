"""Inception Score (IS) for every model evaluated by evaluate.py, with two classifiers.

1. inception_score_cub - the CUB-paper IS (StackGAN, AttnGAN, DM-GAN, DF-GAN, SSA-GAN, RAT-GAN): StackGAN's
   Inception-v3 fine-tuned on CUB (50 classes), ported to PyTorch in inception_cub.py
   (pretrained/inception_cub.pth; see that file to create it). Computed from the cached images
   (cache/gen_<tag>.npy, cache/real_test_256.npy), resized 256 -> 299, 10 splits, mean +- std.
   Papers use ~30k images (10 per test image): run evaluate.py --stage gen with EVAL_CAPS_PER_IMAGE=10 and a
   tag ending in "_x10" for that protocol; those results go to results/paper_protocol/.
2. inception_score - ImageNet Inception-v3 (pytorch-fid's port of TF "inception-2015-12-05"), from the cached
   pool3 features (cache/feats_<tag>.pt), unbiased logits as in OpenAI's improved-GAN code. Kept for
   continuity with our earlier tables; not comparable with paper values.

Usage: python evaluation/inception_score.py [--cache evaluation/cache] [--results evaluation/results]
"""
import argparse, fnmatch, glob, hashlib, json, os, sys
from pathlib import Path

import numpy as np
import torch
from pytorch_fid.inception import FID_WEIGHTS_URL
from torch.hub import get_dir, load_state_dict_from_url

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import inception_cub  # noqa: E402

IMAGENET_PROTOCOL = "ImageNet Inception-v3 (TF 2015-12-05 weights), unbiased logits, 10 splits"


def fc_weight():
    sd = load_state_dict_from_url(FID_WEIGHTS_URL, progress=False, map_location="cpu")
    # the file name carries the first 8 hex digits of its sha256; check it (this PC corrupts files)
    path = Path(get_dir()) / "checkpoints" / FID_WEIGHTS_URL.rsplit("/", 1)[1]
    want = path.stem.rsplit("-", 1)[1]
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    assert got.startswith(want), f"{path} is corrupted (sha256 {got[:8]} != {want}); delete it and rerun"
    return sd["fc.weight"].double()                        # 1008 x 2048


def inception_score(pool3, W, splits=10):
    logp = torch.log_softmax(pool3.double() @ W.T, dim=1)  # unbiased logits, as in OpenAI's code
    # The cached images follow the test set order, which is sorted by bird class. Splitting that
    # order gives each split only ~20 classes and penalises text-faithful models, so shuffle first.
    logp = logp[torch.randperm(len(logp), generator=torch.Generator().manual_seed(0))]
    scores, n = [], len(logp)
    for k in range(splits):
        lp = logp[k * n // splits:(k + 1) * n // splits]
        log_py = torch.logsumexp(lp, 0) - np.log(len(lp))  # log of the marginal p(y)
        kl = (lp.exp() * (lp - log_py)).sum(1).mean()
        scores.append(float(kl.exp()))
    return float(np.mean(scores)), float(np.std(scores))


def cub_probabilities(model, npy, cache, device):
    """CUB class probabilities of one cached image set, cached next to it (cubprobs_<tag>.pt)."""
    tag = "real_images" if npy.name == "real_test_256.npy" else npy.stem[len("gen_"):]
    out = cache / f"cubprobs_{tag}.pt"
    if out.is_file() and out.stat().st_mtime > npy.stat().st_mtime:
        return tag, torch.load(out, weights_only=True)
    imgs = np.load(npy, mmap_mode="r")
    print(f"  CUB Inception on {tag} ({len(imgs)} images) ...", flush=True)
    p = inception_cub.probabilities(model, imgs, device)
    torch.save(p, out)
    return tag, p


def update(path, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    res = json.load(open(path)) if path.is_file() else {"tag": path.stem}
    res.update(fields)
    json.dump(res, open(path, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.environ.get("EVAL_CACHE", str(HERE / "cache")))
    ap.add_argument("--results", default=os.environ.get("EVAL_RESULTS", str(HERE / "results")))
    ap.add_argument("--cub_weights", default=str(inception_cub.DEFAULT_WEIGHTS))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tags", nargs="*", default=["*"], help="only these tags (shell patterns, e.g. blm_sgan_*)")
    a = ap.parse_args()
    cache, results = Path(a.cache), Path(a.results)
    wanted = lambda tag: tag == "real_images" or any(fnmatch.fnmatch(tag, t) for t in a.tags)  # noqa: E731

    # 1. CUB-fine-tuned Inception (the paper metric)
    if Path(a.cub_weights).is_file():
        model = inception_cub.load(a.cub_weights, a.device)
        sets = ([cache / "real_test_256.npy"] if (cache / "real_test_256.npy").is_file() else []) + \
            [Path(f) for f in sorted(glob.glob(str(cache / "gen_*.npy"))) if wanted(Path(f).stem[len("gen_"):])]
        for npy in sets:
            tag, p = cub_probabilities(model, npy, cache, a.device)
            m, s = inception_cub.inception_score(p)
            m3, s3 = inception_cub.inception_score(p, splits=3)   # SSA-GAN's IS.py default split count
            dest = results / "paper_protocol" if tag.endswith("_x10") else results
            if tag.endswith("_x10") or (dest / f"{tag}.json").is_file():
                update(dest / f"{tag}.json", {
                    "inception_score_cub": [m, s], "inception_score_cub_n_images": len(p),
                    "inception_score_cub_3splits": [m3, s3],
                    "inception_score_cub_protocol": f"StackGAN CUB-fine-tuned Inception-v3 (50 classes), {len(p)} "
                                                    "images resized 256->299 (bilinear), 10 splits"})
            print(f"{tag:32s} IS (CUB Inception) {m:6.3f} +- {s:.3f}   n={len(p)}")
        del model
    else:
        print(f"no {a.cub_weights}: skipping the CUB-paper IS (see evaluation/inception_cub.py)")

    # 2. ImageNet Inception (from the cached pool3 features)
    W = fc_weight()
    rows = []
    real = cache / "real_reference.pt"
    if real.is_file():
        ref = torch.load(real, weights_only=False)        # written by evaluate.py
        rows.append(("real_images", inception_score(torch.as_tensor(ref["feats"]), W)))
    for f in sorted(glob.glob(str(cache / "feats_*.pt"))):
        if wanted(Path(f).stem[len("feats_"):]):
            fe = torch.load(f, weights_only=True)[0]
            rows.append((Path(f).stem[len("feats_"):], inception_score(fe, W)))
    for tag, (m, s) in rows:
        p = (results / "paper_protocol" if tag.endswith("_x10") else results) / f"{tag}.json"
        if p.is_file():
            update(p, {"inception_score": [m, s], "inception_score_protocol": IMAGENET_PROTOCOL})
        print(f"{tag:32s} IS (ImageNet Inception) {m:6.3f} +- {s:.3f}" + ("" if p.is_file() else "   (no results json)"))


if __name__ == "__main__":
    main()
