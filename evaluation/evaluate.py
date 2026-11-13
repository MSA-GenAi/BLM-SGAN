"""Evaluate a text-to-image generator on the CUB test split.

Metrics (same protocol for every model):
  * FID            - Inception-v3 pool3 features (pytorch-fid weights), generated vs the 2,933
                     bbox-cropped real test images.            lower = better image quality
                     fid_uncropped: the same against the whole test photos (SSA-GAN's reported setup).
  * R-precision    - for each image, rank its true caption against 99 random test captions
                     (top-1 accuracy, mean +- std over 10 draws).   higher = better text match
                     judged by two models: DAMSM (AttnGAN encoders, the standard CUB metric) and
                     CLIP ViT-B/32 (independent of every model being compared).
  * CLIP score     - mean cosine similarity between CLIP ViT-B/32 embeddings of each image and its full caption
                     (as reported by LAFITE / GALIP).                        higher = better text match
  * KID            - Kernel Inception Distance (Binkowski et al. 2018): unbiased MMD^2 with a cubic polynomial
                     kernel on the FID features, 100 subsets of 1,000.      lower = better
  * FD_CLIP        - Frechet distance on CLIP ViT-B/32 image embeddings (Kynkaanniemi et al. 2023: less tied to
                     ImageNet classes than FID).                             lower = better
  * Precision / Recall (Kynkaanniemi et al. 2019) and Density / Coverage (Naeem et al. 2020), k = 5 nearest
                     neighbours on the FID features, equal real/generated counts.   fidelity vs diversity

Usage:
  python evaluation/evaluate.py --model ssagan --ckpt <netG.pth> --tag ssagan_590
  python evaluation/evaluate.py --model blm    --ckpt checkpoints/birds/blm_sgan/netG_555.pth --tag blm_sgan_555
  python evaluation/evaluate.py --real                       (real-image reference values)

First run: python evaluation/prepare_real.py --cub <CUB_200_2011.tgz or the extracted CUB_200_2011 folder>
"""
import argparse, json, os, pickle, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "birds"                      # captions.pickle, test/filenames.pickle, test/class_info.pickle
DAMSM_DIR = REPO / "DAMSMencoders" / "bird"         # text_encoder200.pth, image_encoder200.pth (AttnGAN)
CLIP_PATH = os.environ.get("CLIP_PATH", "openai/clip-vit-base-patch32")
HERE = Path(__file__).resolve().parent
# EVAL_CACHE / EVAL_RESULTS move them (e.g. on Colab: cache on the local disk, results on Google Drive)
CACHE = Path(os.environ.get("EVAL_CACHE", HERE / "cache"))
RESULTS = Path(os.environ.get("EVAL_RESULTS", HERE / "results"))
CACHE.mkdir(exist_ok=True); RESULTS.mkdir(exist_ok=True)
sys.path.insert(0, str(REPO))
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import types  # noqa: E402
import DAMSM as damsm_mod  # noqa: E402
# The DAMSM image encoder state dict contains every Inception layer, so skip the ImageNet download
# (patched only inside DAMSM.py; pytorch-fid keeps the real torchvision function).
_tv_inception = torchvision.models.inception_v3
damsm_mod.models = types.SimpleNamespace(inception_v3=lambda pretrained=False, transform_input=False: _tv_inception(
    weights=None, aux_logits=True, init_weights=False, transform_input=transform_input))
from DAMSM import RNN_ENCODER, CNN_ENCODER  # noqa: E402

META = pickle.load(open(DATA / "captions.pickle", "rb"))
TEST_CAPS, IXTOWORD = META[1], META[2]
N_IMG = len(TEST_CAPS) // 10
CAP_PER_IMG = int(os.environ.get("EVAL_CAPS_PER_IMAGE", "1"))
PAIRS = [(i, i * 10 + (i * 7 + k * 3) % 10) for i in range(N_IMG) for k in range(CAP_PER_IMG)]  # (image, caption)
CAP_TEXT = [" ".join(IXTOWORD[j] for j in c[:18]) for c in TEST_CAPS]   # R-precision: 18 words, as DAMSM
CAP_FULL = [" ".join(IXTOWORD[j] for j in c) for c in TEST_CAPS]         # CLIP score: the whole caption


# ----------------------------------------------------------------------------- judges
class Judges:
    def __init__(self):
        from pytorch_fid.inception import InceptionV3
        from transformers import CLIPModel, CLIPTokenizer
        self.fid = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(dev).eval()
        self.img_enc = CNN_ENCODER(256)
        self.img_enc.load_state_dict(torch.load(DAMSM_DIR / "image_encoder200.pth", map_location="cpu", weights_only=True))
        self.img_enc.to(dev).eval()
        self.rnn = RNN_ENCODER(len(IXTOWORD), nhidden=256)
        self.rnn.load_state_dict(torch.load(DAMSM_DIR / "text_encoder200.pth", map_location="cpu", weights_only=True))
        self.rnn.to(dev).eval()
        self.clip = CLIPModel.from_pretrained(CLIP_PATH, use_safetensors=True).to(dev).eval()
        self.clip_tok = CLIPTokenizer.from_pretrained(CLIP_PATH)
        self.clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=dev).view(1, 3, 1, 1)
        self.clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=dev).view(1, 3, 1, 1)

    @torch.no_grad()
    def images(self, x):
        """x: B x 3 x 256 x 256 in [-1, 1] -> (fid feats, damsm code, clip emb)."""
        x01 = (x.clamp(-1, 1) + 1) / 2
        f = self.fid(x01)[0].squeeze(-1).squeeze(-1)
        _, d = self.img_enc(x.clamp(-1, 1))
        c = F.interpolate(x01, size=(224, 224), mode="bicubic", align_corners=False).clamp(0, 1)
        # = get_image_features(), spelled out: transformers 5 returns an output object from get_*_features()
        c = self.clip.visual_projection(self.clip.vision_model(pixel_values=(c - self.clip_mean) / self.clip_std).pooler_output)
        return f.double().cpu(), F.normalize(d, dim=1).cpu(), F.normalize(c, dim=1).cpu()

    @torch.no_grad()
    def clip_text(self, texts):
        cl = []
        for s in range(0, len(texts), 512):
            t = self.clip_tok(texts[s:s + 512], padding=True, truncation=True, max_length=77, return_tensors="pt").to(dev)
            cl.append(self.clip.text_projection(self.clip.text_model(**t).pooler_output).cpu())   # = get_text_features()
        return F.normalize(torch.cat(cl), dim=1)

    @torch.no_grad()
    def text_tables(self):
        p = CACHE / "text_tables.pt"
        out = torch.load(p, weights_only=True) if p.is_file() else {}
        changed = False
        if "damsm" not in out:
            out["damsm"] = F.normalize(torch.cat([rnn_encode(self.rnn, TEST_CAPS[s:s + 512]).cpu()
                                                  for s in range(0, len(TEST_CAPS), 512)]), dim=1)
            out["clip"] = self.clip_text(CAP_TEXT)
            changed = True
        if "clip_full" not in out:
            out["clip_full"] = self.clip_text(CAP_FULL)
            changed = True
        if changed:
            torch.save(out, p)
        return out


def rnn_encode(rnn, id_lists):
    lens = torch.tensor([max(1, min(len(c), 18)) for c in id_lists])
    order = torch.argsort(lens, descending=True)
    rows = torch.zeros(len(id_lists), 18, dtype=torch.long)
    for k, c in enumerate(id_lists):
        c = list(c)[:18]
        rows[k, :len(c)] = torch.tensor(c, dtype=torch.long)
    _, s = rnn(rows[order].to(dev), lens[order].to(dev), rnn.init_hidden(len(id_lists)))
    inv = torch.empty_like(order); inv[order] = torch.arange(len(order))
    return s[inv.to(dev)]


def r_precision(img_emb, cap_idx, table, draws=10, cands=100):
    """Top-1 accuracy of the true caption among `cands` candidates (true + others from different images)."""
    n = len(cap_idx)
    true = (img_emb * table[cap_idx]).sum(1)
    accs = []
    for d in range(draws):
        g = torch.Generator().manual_seed(1000 + d)
        other = torch.randint(0, len(table), (n, cands - 1), generator=g)
        same_img = (other // 10) == (torch.tensor(cap_idx) // 10).unsqueeze(1)
        other[same_img] = (other[same_img] + 10) % len(table)          # never a caption of the same image
        sims = torch.einsum("nd,nkd->nk", img_emb, table[other])
        accs.append((true > sims.max(1).values).float().mean().item())
    return float(np.mean(accs) * 100), float(np.std(accs) * 100)


def frechet_distance(mu1, s1, mu2, s2):
    """FID = |mu1-mu2|^2 + tr(S1) + tr(S2) - 2 tr(sqrt(S1 S2)).

    tr(sqrt(S1 S2)) is computed as tr(sqrt(S1^1/2 S2 S1^1/2)), which is mathematically identical but only needs
    symmetric eigendecompositions (fast and stable). scipy.linalg.sqrtm on the raw product takes many minutes
    on real, ill-conditioned covariances with recent SciPy versions.
    """
    from scipy import linalg
    s1, s2 = np.asarray(s1, np.float64), np.asarray(s2, np.float64)
    assert np.isfinite(s1).all() and np.isfinite(s2).all(), "non-finite covariance"
    diff = mu1 - mu2

    def psd_sqrt(s):
        w, v = linalg.eigh((s + s.T) / 2, driver="evr")
        return (v * np.sqrt(np.clip(w, 0, None))) @ v.T

    r1, r2 = psd_sqrt(s1), psd_sqrt(s2)
    # tr sqrt(S1^1/2 S2 S1^1/2) = nuclear norm of S1^1/2 S2^1/2  (sum of singular values)
    tr_covmean = linalg.svdvals(r1 @ r2).sum()
    return float(diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2 * tr_covmean)


def fid_from(feats, ref):
    mu, sig = feats.mean(0).numpy(), np.cov(feats.numpy(), rowvar=False)
    return frechet_distance(mu, sig, ref["mu"], ref["sigma"])


def fd(a, b):
    """Frechet distance between two feature sets (rows = samples)."""
    a, b = a.double().numpy(), b.double().numpy()
    return frechet_distance(a.mean(0), np.cov(a, rowvar=False), b.mean(0), np.cov(b, rowvar=False))


def kid(fake, real, subsets=100, size=1000, seed=0):
    """Kernel Inception Distance: unbiased MMD^2, kernel k(x,y) = (x.y/d + 1)^3 (torch-fidelity defaults)."""
    fake, real = fake.double(), real.double()
    d, m = fake.shape[1], min(size, len(fake), len(real))
    g = torch.Generator().manual_seed(seed)
    vals = []
    for _ in range(subsets):
        x = fake[torch.randperm(len(fake), generator=g)[:m]]
        y = real[torch.randperm(len(real), generator=g)[:m]]
        kxx, kyy, kxy = (x @ x.T / d + 1) ** 3, (y @ y.T / d + 1) ** 3, (x @ y.T / d + 1) ** 3
        vals.append(((kxx.sum() - kxx.diagonal().sum()) / (m * (m - 1))
                     + (kyy.sum() - kyy.diagonal().sum()) / (m * (m - 1)) - 2 * kxy.mean()).item())
    return float(np.mean(vals)), float(np.std(vals))


def prdc(fake, real, k=5, seed=0):
    """Precision, recall (Kynkaanniemi et al. 2019), density, coverage (Naeem et al. 2020); same formulas as
    the reference `prdc` package. A larger generated set is subsampled to the size of the real set."""
    fake, real = fake.double(), real.double()
    if len(fake) > len(real):
        fake = fake[torch.randperm(len(fake), generator=torch.Generator().manual_seed(seed))[:len(real)]]
    r_rad = torch.cdist(real, real).kthvalue(k + 1, dim=1).values      # k-th neighbour (the 1st is the point itself)
    f_rad = torch.cdist(fake, fake).kthvalue(k + 1, dim=1).values
    d_rf = torch.cdist(real, fake)                                       # real x fake
    inside = d_rf < r_rad[:, None]
    return {"precision": inside.any(0).double().mean().item(),
            "recall": (d_rf < f_rad[None, :]).any(1).double().mean().item(),
            "density": (inside.sum(0).double() / k).mean().item(),
            "coverage": (d_rf.min(1).values < r_rad).double().mean().item()}


def extra_metrics(fe, ce, ref, clip_text):
    """KID, FD_CLIP, precision/recall/density/coverage and CLIP score of one generated set."""
    out = {"kid": kid(fe, ref["feats"]), "fd_clip": fd(ce, ref["clip"])}
    out.update(prdc(fe, ref["feats"]))
    out["clip_score"] = float((ce * clip_text).sum(1).mean())
    return out


# ----------------------------------------------------------------------------- generators
def make_generator(kind, ckpt, encoding=None):
    """kind "blm": BLM-SGAN, the full caption text -> its BERT-DAMSM encoder (text_encoder.pth next to the generator);
    kind "ssagan": SSA-GAN, the caption's vocabulary ids (first 18 words) -> AttnGAN's DAMSM LSTM."""
    if kind == "blm":
        from model import NetG
        sys.path.insert(0, str(REPO / "training"))
        from text_encoder import BertTextEncoder
        G = NetG(64, 100)
        sd = torch.load(ckpt, map_location="cpu", weights_only=True)
        G.load_state_dict({k[7:] if k.startswith("module.") else k: v for k, v in sd.items()})
        G.to(dev).eval()
        bert = REPO / "pretrained" / "bert-base-uncased"
        txt = BertTextEncoder.load(Path(ckpt).parent / "text_encoder.pth",
                                   bert=str(bert) if (bert / "config.json").is_file() else None).to(dev).eval()
        return lambda ids, z: G(z, txt([" ".join(IXTOWORD[j] for j in c) for c in ids])[1])[0]
    if kind == "ssagan":
        from model import NetG
        G = NetG(64, 100)
        sd = torch.load(ckpt, map_location="cpu", weights_only=True)
        G.load_state_dict({k[7:] if k.startswith("module.") else k: v for k, v in sd.items()})
        G.to(dev).eval()
        rnn = RNN_ENCODER(len(IXTOWORD), nhidden=256)
        rnn.load_state_dict(torch.load(DAMSM_DIR / "text_encoder200.pth", map_location="cpu", weights_only=True))
        rnn.to(dev).eval()
        return lambda ids, z: G(z, rnn_encode(rnn, ids))[0]
    raise ValueError(f"unknown model kind {kind!r}")


# ----------------------------------------------------------------------------- main
def uncropped_reference(J):
    """FID statistics of the whole (uncropped) test photos, or None before prepare_real.py made them."""
    p, src = CACHE / "real_reference_uncropped.pt", CACHE / "real_test_256_uncropped.npy"
    if p.is_file():
        return torch.load(p, weights_only=False)
    if not src.is_file():
        return None
    real = np.load(src)
    fe = torch.cat([J.images(torch.from_numpy(real[s:s + 32]).permute(0, 3, 1, 2).float().to(dev) / 127.5 - 1)[0]
                    for s in range(0, len(real), 32)])
    ref = {"mu": fe.mean(0).numpy(), "sigma": np.cov(fe.numpy(), rowvar=False)}
    torch.save(ref, p)
    return ref


def real_reference(J):
    p = CACHE / "real_reference.pt"
    if p.is_file():
        return torch.load(p, weights_only=False)
    real = np.load(CACHE / "real_test_256.npy")
    fe, de, ce = [], [], []
    for s in range(0, len(real), 32):
        x = torch.from_numpy(real[s:s + 32]).permute(0, 3, 1, 2).float().to(dev) / 127.5 - 1
        f, d, c = J.images(x)
        fe.append(f); de.append(d); ce.append(c)
    fe, de, ce = torch.cat(fe), torch.cat(de), torch.cat(ce)
    ref = {"mu": fe.mean(0).numpy(), "sigma": np.cov(fe.numpy(), rowvar=False), "damsm": de, "clip": ce, "feats": fe}
    torch.save(ref, p)
    return ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["ssagan", "blm"])
    ap.add_argument("--ckpt")
    ap.add_argument("--tag")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--stage", default="all", choices=["gen", "judge", "all"],
                    help="gen: only generate images to cache; judge: only score cached images")
    a = ap.parse_args()
    t0 = time.time()
    cap_i = [p[1] for p in PAIRS]
    gen_file = CACHE / f"gen_{a.tag}.npy"
    if not a.real and a.stage in ("gen", "all"):
        gen = make_generator(a.model, a.ckpt)
        out = np.lib.format.open_memmap(gen_file, mode="w+", dtype=np.uint8, shape=(len(PAIRS), 256, 256, 3))
        zgen = torch.Generator().manual_seed(2024)
        for s in range(0, len(PAIRS), a.bs):
            idx = cap_i[s:s + a.bs]
            z = torch.randn(len(idx), 100, generator=zgen).to(dev)
            with torch.no_grad():
                x = gen([TEST_CAPS[i] for i in idx], z)
            out[s:s + len(idx)] = ((x.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
            if s % (a.bs * 40) == 0:
                print(f"  generated {s}/{len(PAIRS)}  {time.time() - t0:.0f}s", flush=True)
        out.flush(); del out
        json.dump({"tag": a.tag, "model": a.model, "ckpt": str(a.ckpt), "encoding": {"blm": "bert_damsm"}.get(a.model, "damsm_lstm")},
                  open(CACHE / f"gen_{a.tag}.json", "w"))
        print(f"  generation done in {time.time() - t0:.0f}s", flush=True)
        if a.stage == "gen":
            return
    J = Judges()
    tables = J.text_tables()
    ref = real_reference(J)
    ref_full = uncropped_reference(J)
    if a.real:
        img_i = [p[0] for p in PAIRS]; cap_i = [p[1] for p in PAIRS]
        perm = torch.randperm(len(ref["feats"]), generator=torch.Generator().manual_seed(0))
        h1, h2 = perm[:len(perm) // 2], perm[len(perm) // 2:]     # two random halves (the files are in class order)
        res = {"tag": "real_images",
               "fid_real_half_vs_half": fd(ref["feats"][h1], ref["feats"][h2]),
               "r_precision_damsm": r_precision(ref["damsm"][img_i], cap_i, tables["damsm"]),
               "r_precision_clip": r_precision(ref["clip"][img_i], cap_i, tables["clip"]),
               "clip_score": float((ref["clip"][img_i] * tables["clip_full"][cap_i]).sum(1).mean()),
               "half_vs_half": {"kid": kid(ref["feats"][h1], ref["feats"][h2]),
                                "fd_clip": fd(ref["clip"][h1], ref["clip"][h2]),
                                **prdc(ref["feats"][h1], ref["feats"][h2])}}
        old = RESULTS / "real_images.json"
        res = {**(json.load(open(old)) if old.is_file() else {}), **res}
        print(json.dumps(res, indent=1)); json.dump(res, open(RESULTS / "real_images.json", "w"), indent=1)
        return
    imgs = np.load(gen_file, mmap_mode="r")
    meta = json.load(open(CACHE / f"gen_{a.tag}.json"))
    feat_file = CACHE / f"feats_{a.tag}.pt"
    if feat_file.is_file():
        fe, de, ce = torch.load(feat_file, weights_only=True)
    else:
        fe, de, ce = [], [], []
        for s in range(0, len(imgs), a.bs):
            x = torch.from_numpy(np.ascontiguousarray(imgs[s:s + a.bs])).permute(0, 3, 1, 2).float().to(dev) / 127.5 - 1
            f, d, c = J.images(x)
            fe.append(f); de.append(d); ce.append(c)
            if s % (a.bs * 40) == 0:
                print(f"  judged {s}/{len(imgs)}  {time.time() - t0:.0f}s", flush=True)
        fe, de, ce = torch.cat(fe), torch.cat(de), torch.cat(ce)
        torch.save((fe, de, ce), feat_file)
    grid = np.asarray(imgs[:16])
    print("  computing FID / R-precision / KID / precision-recall / CLIP score", flush=True)
    res = {**meta, "n_images": len(imgs), "fid": fid_from(fe, ref),
           **({"fid_uncropped": fid_from(fe, ref_full)} if ref_full is not None else {}),
           "r_precision_damsm": r_precision(de, cap_i, tables["damsm"]),
           "r_precision_clip": r_precision(ce, cap_i, tables["clip"]),
           **extra_metrics(fe, ce, ref, tables["clip_full"][cap_i]),
           "seconds": round(time.time() - t0)}
    dest = RESULTS / "paper_protocol" if a.tag.endswith("_x10") else RESULTS    # same rule as inception_score.py
    dest.mkdir(exist_ok=True)
    from PIL import Image
    Image.fromarray(np.concatenate([np.concatenate(list(grid[r * 4:(r + 1) * 4]), 1) for r in range(4)], 0)).save(dest / f"{a.tag}_samples.png")
    with open(dest / f"{a.tag}_captions.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(CAP_TEXT[i] for i in cap_i[:16]))
    old = dest / f"{a.tag}.json"     # keep fields other scripts added (Inception Scores, labels)
    res = {**(json.load(open(old)) if old.is_file() else {}), **res}
    print(json.dumps(res, indent=1)); json.dump(res, open(old, "w"), indent=1)


if __name__ == "__main__":
    main()
    # Every result file is written and closed at this point. Skip the interpreter teardown: on Windows it
    # sometimes segfaults (CUDA + transformers threads), which made a finished evaluation look failed.
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)
