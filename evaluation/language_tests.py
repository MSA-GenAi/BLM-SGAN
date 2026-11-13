"""Two language tests for BLM-SGAN (BERT text encoder) vs SSA-GAN (AttnGAN's DAMSM LSTM).

They test what a pretrained language model should add. Both were fixed before running and are reported
whatever the outcome. Every image is judged by CLIP ViT-B/32, which neither generator was trained with.

1. Rare words. Each of the 2,933 test captions of the main protocol has its colour and size words replaced by
   synonyms that SSA-GAN's text encoder saw at most twice in its 88,550 training captions. Words never seen
   are not in its vocabulary at all and are dropped, as SSA-GAN's own code does. BERT knows all of them from
   pre-training. For each model we generate, with the same noise:
     orig = the original caption
     rare = the caption with the synonyms
     rand = the caption of another test image (no information about this bird)
   and score each image against the ORIGINAL caption with:
     * CLIP score;
     * CLIP R-precision, where the original caption ranks first among 100.
   kept = (rare - rand) / (orig - rand) is the share of the caption's effect that survives the rare words.
2. Long captions. SSA-GAN reads only the first 18 words of a caption, while BLM-SGAN reads up to 40 BERT tokens.
   For test captions longer than 18 words that fit BLM-SGAN's window (one per image), images come from:
     * SSA-GAN;
     * BLM-SGAN given the first 18 words;
     * BLM-SGAN given the whole caption.
   They are scored with:
     * CLIP score against the words after the 18th (the tail);
     * tail R-precision (own tail first among 100 tails);
     * CLIP score against the whole caption.
   "whole - first 18" for BLM-SGAN measures whether the details after word 18 reach the image.

Usage: python evaluation/language_tests.py --blm checkpoints/birds/blm_sgan/netG_555.pth --tag blm_sgan_555
       [--ssagan checkpoints/birds/ssagan/netG_590.pth] [--bs 16] [--limit N]
Writes evaluation/results/language_tests/<tag>.json (results folder: EVAL_RESULTS).
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate as E  # noqa: E402  (captions, vocabulary, DAMSM LSTM, CLIP settings)

dev = E.dev
WORDS_NUM = 18          # SSA-GAN / AttnGAN cfg.TEXT.WORDS_NUM
MAX_TRAIN_COUNT = 2     # a synonym qualifies if SSA-GAN's training captions contain it at most this often
SYNONYMS = {            # chosen before any image was generated; the counts are checked again at run time
    "red": "vermilion", "blue": "ultramarine", "yellow": "saffron", "black": "ebony", "white": "alabaster",
    "brown": "umber", "orange": "tangerine", "grey": "ashen", "gray": "ashen", "green": "jade",
    "small": "diminutive", "large": "sizable",
}


# ----------------------------------------------------------------------------- generators
def load_netg(ckpt):
    from model import NetG
    G = NetG(64, 100)
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    G.load_state_dict({k[7:] if k.startswith("module.") else k: v for k, v in sd.items()})
    return G.to(dev).eval()


class SSAGAN:
    """Caption text -> CUB vocabulary ids (unknown words dropped, as datasets.py does) -> DAMSM LSTM (18 words)."""
    def __init__(self, ckpt):
        self.G = load_netg(ckpt)
        self.rnn = E.RNN_ENCODER(len(E.IXTOWORD), nhidden=256)
        self.rnn.load_state_dict(torch.load(E.DAMSM_DIR / "text_encoder200.pth", map_location="cpu", weights_only=True))
        self.rnn.to(dev).eval()
        self.w2i = {w: i for i, w in E.IXTOWORD.items()}

    @torch.no_grad()
    def __call__(self, texts, z):
        ids = [[self.w2i[w] for w in t.split() if w in self.w2i] or [0] for t in texts]
        return self.G(z, E.rnn_encode(self.rnn, ids))[0]


class BLMSGAN:
    """Caption text -> BLM-SGAN's BERT-DAMSM text encoder (text_encoder.pth next to the generator)."""
    def __init__(self, ckpt):
        sys.path.insert(0, str(E.REPO / "training"))
        from text_encoder import BertTextEncoder
        self.G = load_netg(ckpt)
        bert = E.REPO / "pretrained" / "bert-base-uncased"
        self.txt = BertTextEncoder.load(Path(ckpt).parent / "text_encoder.pth",
                                        bert=str(bert) if (bert / "config.json").is_file() else None).to(dev).eval()

    @torch.no_grad()
    def __call__(self, texts, z):
        return self.G(z, self.txt(list(texts))[1])[0]


class Clip:
    """CLIP ViT-B/32 with the same preprocessing as evaluate.Judges."""
    def __init__(self):
        from transformers import CLIPModel, CLIPTokenizer
        self.m = CLIPModel.from_pretrained(E.CLIP_PATH, use_safetensors=True).to(dev).eval()
        self.tok = CLIPTokenizer.from_pretrained(E.CLIP_PATH)
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=dev).view(1, 3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=dev).view(1, 3, 1, 1)

    @torch.no_grad()
    def images(self, x):
        x01 = (x.clamp(-1, 1) + 1) / 2
        c = F.interpolate(x01, size=(224, 224), mode="bicubic", align_corners=False).clamp(0, 1)
        c = self.m.visual_projection(self.m.vision_model(pixel_values=(c - self.mean) / self.std).pooler_output)
        return F.normalize(c, dim=1).cpu()

    @torch.no_grad()
    def texts(self, texts):
        out = []
        for s in range(0, len(texts), 256):
            t = self.tok(texts[s:s + 256], padding=True, truncation=True, max_length=77, return_tensors="pt").to(dev)
            out.append(self.m.text_projection(self.m.text_model(**t).pooler_output).cpu())
        return F.normalize(torch.cat(out), dim=1)


def embed(gen, clip, texts, z, bs, label):
    t0, out = time.time(), []
    for s in range(0, len(texts), bs):
        out.append(clip.images(gen(texts[s:s + bs], z[s:s + bs].to(dev))))
    print(f"  {label:28s} {len(texts)} images  {time.time() - t0:.0f}s", flush=True)
    return torch.cat(out)


# ----------------------------------------------------------------------------- statistics
def mean_se(v):
    v = v.double()
    return [float(v.mean()), float(v.std() / len(v) ** 0.5)]


def r_prec(img, txt, groups, draws=10, cands=100):
    """Top-1 accuracy of row i of `txt` for image i among `cands` rows (others from a different group/image)."""
    n, groups = len(img), torch.as_tensor(groups)
    true = (img * txt).sum(1)
    accs = []
    for d in range(draws):
        g = torch.Generator().manual_seed(1000 + d)
        other = torch.randint(0, n, (n, cands - 1), generator=g)
        clash = groups[other] == groups.unsqueeze(1)
        while clash.any():
            other[clash] = torch.randint(0, n, (int(clash.sum()),), generator=g)
            clash = groups[other] == groups.unsqueeze(1)
        accs.append((true > torch.einsum("nd,nkd->nk", img, txt[other]).max(1).values).float().mean().item())
    return [float(np.mean(accs) * 100), float(np.std(accs) * 100)]


# ----------------------------------------------------------------------------- tests
def rare_word_test(models, clip, bs, limit):
    train_count = {}
    for c in E.META[0]:
        for j in c:
            train_count[E.IXTOWORD[j]] = train_count.get(E.IXTOWORD[j], 0) + 1
    syn = {w: s for w, s in SYNONYMS.items() if train_count.get(s, 0) <= MAX_TRAIN_COUNT}
    assert len(syn) == len(SYNONYMS), f"synonyms too frequent in training: {set(SYNONYMS) - set(syn)}"
    rows = []
    for i, ci in E.PAIRS:
        words = [E.IXTOWORD[j] for j in E.TEST_CAPS[ci]]
        if any(w in syn for w in words):
            rows.append((i, ci, " ".join(words), " ".join(syn.get(w, w) for w in words)))
    rows = rows[:limit] if limit else rows
    n = len(rows)
    by_img = {i: ci for i, ci in E.PAIRS}
    imgs = sorted(by_img)
    rand_txt = [" ".join(E.IXTOWORD[j] for j in E.TEST_CAPS[by_img[imgs[(imgs.index(i) + len(imgs) // 2) % len(imgs)]]])
                for i, _, _, _ in rows]                                     # a caption of an unrelated test image
    orig_txt, rare_txt = [r[2] for r in rows], [r[3] for r in rows]
    replaced = sum(sum(w in syn for w in r[2].split()) for r in rows)
    print(f"rare-word test: {n} captions, {replaced} words replaced", flush=True)
    ref = clip.texts(orig_txt)
    z = torch.randn(n, 100, generator=torch.Generator().manual_seed(2024))
    vocab = set(E.IXTOWORD.values())
    res = {"n_captions": n, "n_words_replaced": replaced, "max_train_count": MAX_TRAIN_COUNT,
           "synonyms": {w: {"synonym": s, "times_in_ssagan_training": train_count.get(s, 0),
                            "in_ssagan_vocabulary": s in vocab} for w, s in syn.items()},
           "examples": [[r[2], r[3]] for r in rows[:5]], "models": {}}
    for name, gen in models.items():
        emb = {k: embed(gen, clip, t, z, bs, f"{name} {k}") for k, t in
               (("orig", orig_txt), ("rare", rare_txt), ("rand", rand_txt))}
        s = {k: (e * ref).sum(1) for k, e in emb.items()}
        o, r_, b = (float(s[k].mean()) for k in ("orig", "rare", "rand"))
        res["models"][name] = {
            "clip_score": {k: mean_se(v) for k, v in s.items()},
            "clip_score_drop": mean_se(s["orig"] - s["rare"]),
            "kept": (r_ - b) / (o - b),
            "r_precision_clip": {k: r_prec(e, ref, [r[0] for r in rows]) for k, e in emb.items()},
        }
    return res


def long_caption_test(models, clip, bs, limit):
    from transformers import BertTokenizer
    bert = E.REPO / "pretrained" / "bert-base-uncased"
    tok = BertTokenizer.from_pretrained(str(bert) if (bert / "config.json").is_file() else "bert-base-uncased")
    rows = []
    for i in range(E.N_IMG):                        # one caption per image: the first longer than 18 words
        for k in range(10):
            words = [E.IXTOWORD[j] for j in E.TEST_CAPS[i * 10 + k]]
            if len(words) > WORDS_NUM and len(tok.tokenize(" ".join(words))) <= 38:   # [CLS] + 38 + [SEP] = 40
                rows.append((i, words))
                break
    rows = rows[:limit] if limit else rows
    n = len(rows)
    full = [" ".join(w) for _, w in rows]
    head = [" ".join(w[:WORDS_NUM]) for _, w in rows]
    tail = [" ".join(w[WORDS_NUM:]) for _, w in rows]
    print(f"long-caption test: {n} captions, {np.mean([len(w) for _, w in rows]):.1f} words on average", flush=True)
    ref_tail, ref_full = clip.texts(tail), clip.texts(full)
    z = torch.randn(n, 100, generator=torch.Generator().manual_seed(2025))
    runs = [(f"{name} (first 18 words)", gen, head) for name, gen in models.items() if name == "ssagan"] + \
           [(f"{name} (first 18 words)", gen, head) for name, gen in models.items() if name != "ssagan"] + \
           [(f"{name} (whole caption)", gen, full) for name, gen in models.items() if name != "ssagan"]
    res = {"n_captions": n, "mean_words": float(np.mean([len(w) for _, w in rows])),
           "examples": [[h, t] for h, t in zip(head[:5], tail[:5])], "runs": {}}
    scores = {}
    for label, gen, texts in runs:
        e = embed(gen, clip, texts, z, bs, label)
        scores[label] = (e * ref_tail).sum(1)
        res["runs"][label] = {"clip_score_tail": mean_se(scores[label]),
                              "r_precision_tail": r_prec(e, ref_tail, [r[0] for r in rows]),
                              "clip_score_full": mean_se((e * ref_full).sum(1))}
    for name in models:
        if name != "ssagan":
            res[f"{name}: whole - first 18 (tail CLIP score)"] = mean_se(
                scores[f"{name} (whole caption)"] - scores[f"{name} (first 18 words)"])
            if "ssagan" in models:
                res[f"{name} whole - ssagan (tail CLIP score)"] = mean_se(
                    scores[f"{name} (whole caption)"] - scores["ssagan (first 18 words)"])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blm", required=True, help="BLM-SGAN netG_<epoch>.pth (text_encoder.pth next to it)")
    ap.add_argument("--ssagan", default=str(E.REPO / "checkpoints" / "birds" / "ssagan" / "netG_590.pth"))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="first N captions only (a quick test)")
    a = ap.parse_args()
    t0 = time.time()
    if sys.platform == "win32":   # no idle sleep while this process runs (a sleeping PC stalled a run for 10 h)
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    clip = Clip()
    models = {"ssagan": SSAGAN(a.ssagan), "blm_sgan": BLMSGAN(a.blm)}
    out = {"tag": a.tag, "blm_sgan": str(a.blm), "ssagan": str(a.ssagan), "judge": E.CLIP_PATH,
           "rare_words": rare_word_test(models, clip, a.bs, a.limit),
           "long_captions": long_caption_test(models, clip, a.bs, a.limit),
           "seconds": round(time.time() - t0)}
    dest = E.RESULTS / "language_tests"
    dest.mkdir(exist_ok=True)
    name = f"{a.tag}{'_limit' + str(a.limit) if a.limit else ''}.json"
    with open(dest / name, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))
    print("wrote", dest / name)


if __name__ == "__main__":
    main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)   # see evaluate.py: skip the Windows CUDA teardown crash
