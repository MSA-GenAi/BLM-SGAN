"""FID exactly as DF-GAN (CVPR 2022) and GALIP (CVPR 2023) compute it, so our numbers can sit in their tables.

Their protocol differs from evaluate.py's (pytorch-fid) in four ways:
  * network:   torchvision's ImageNet Inception-v3 (their models/inception.py), not the TF FID weights;
  * inputs:    images in [0, 1] go through torchvision's `transform_input` formula (meant for ImageNet-normalised
               inputs), so the network sees roughly [-0.03, 0.43]. A quirk, but the reference statistics were made
               the same way, so it has to be copied exactly;
  * reference: DF-GAN's precomputed statistics bird_val256_FIDK0.npz (birds metadata zip of DF-GAN's README,
               https://github.com/tobran/DF-GAN), put at data/birds/npz/;
  * samples:   every test image with one random caption of its 10, repeated --times passes.
               DF-GAN: 10 passes, noise = 0.88 x normal truncated at +-2, captions over 18 words cut to a random
               18-word subset (its dataset loader). GALIP: 12 passes, plain normal noise.
The DF-GAN generator is re-implemented below (architecture of DF-GAN's code/models/GAN.py, CC BY-NC-SA 4.0) only to
check this script against DF-GAN's released bird model (README: FID 12.10).

Usage:
  python evaluation/fid_dfgan.py --real                     (our real test images against the npz)
  python evaluation/fid_dfgan.py --model dfgan --ckpt <state_epoch_1220.pth> --tag dfgan_released
  python evaluation/fid_dfgan.py --model ssagan --ckpt checkpoints/birds/ssagan/netG_590.pth --tag ssagan_590
  python evaluation/fid_dfgan.py --model blm --ckpt checkpoints/birds/blm_sgan/netG_555.pth --tag blm_sgan_555_dfgan
  (add --protocol galip --tag ..._galip for GALIP's variant)
Results: evaluation/results/dfgan_protocol/<tag>.json
"""
import argparse, hashlib, json, os, sys, time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evaluate as E  # noqa: E402  (captions, generators, real-image cache, Frechet distance)

NPZ = Path(os.environ.get("DFGAN_NPZ", E.REPO / "data" / "birds" / "npz" / "bird_val256_FIDK0.npz"))
OUT = E.RESULTS / "dfgan_protocol"
dev = E.dev


class DFGANInception(nn.Module):
    """DF-GAN / GALIP models/inception.py: torchvision Inception-v3 up to the 2048-d average pool, fed [0, 1] images
    through torchvision's `transform_input` formula (their input scaling), after bilinear resizing to 299."""

    def __init__(self):
        super().__init__()
        net = torchvision.models.inception_v3(weights=torchvision.models.Inception_V3_Weights.IMAGENET1K_V1)
        self.blocks = nn.Sequential(
            net.Conv2d_1a_3x3, net.Conv2d_2a_3x3, net.Conv2d_2b_3x3, nn.MaxPool2d(3, 2),
            net.Conv2d_3b_1x1, net.Conv2d_4a_3x3, nn.MaxPool2d(3, 2),
            net.Mixed_5b, net.Mixed_5c, net.Mixed_5d, net.Mixed_6a, net.Mixed_6b, net.Mixed_6c, net.Mixed_6d,
            net.Mixed_6e, net.Mixed_7a, net.Mixed_7b, net.Mixed_7c, nn.AdaptiveAvgPool2d(1))
        std, mean = torch.tensor([0.229, 0.224, 0.225]), torch.tensor([0.485, 0.456, 0.406])
        self.register_buffer("scale", (std / 0.5).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("shift", ((mean - 0.5) / 0.5).view(1, 3, 1, 1), persistent=False)

    @torch.no_grad()
    def forward(self, x01):
        """x01: B x 3 x H x W in [0, 1] -> B x 2048 (float64, CPU)."""
        x = F.interpolate(x01.float(), size=(299, 299), mode="bilinear", align_corners=False)   # transforms.Resize
        return self.blocks(x * self.scale + self.shift).flatten(1).double().cpu()


# ----------------------------------------------------------------------------- DF-GAN generator (for the check)
class _Affine(nn.Module):
    def __init__(self, cond_dim, n):
        super().__init__()
        self.fc_gamma = nn.Sequential(OrderedDict([("linear1", nn.Linear(cond_dim, n)), ("relu1", nn.ReLU()),
                                                   ("linear2", nn.Linear(n, n))]))
        self.fc_beta = nn.Sequential(OrderedDict([("linear1", nn.Linear(cond_dim, n)), ("relu1", nn.ReLU()),
                                                  ("linear2", nn.Linear(n, n))]))

    def forward(self, x, y):
        return self.fc_gamma(y)[..., None, None] * x + self.fc_beta(y)[..., None, None]


class _DFBLK(nn.Module):
    def __init__(self, cond_dim, n):
        super().__init__()
        self.affine0, self.affine1 = _Affine(cond_dim, n), _Affine(cond_dim, n)

    def forward(self, x, y):
        return F.leaky_relu(self.affine1(F.leaky_relu(self.affine0(x, y), 0.2), y), 0.2)


class _GBlock(nn.Module):
    def __init__(self, cond_dim, cin, cout):
        super().__init__()
        self.c1, self.c2 = nn.Conv2d(cin, cout, 3, 1, 1), nn.Conv2d(cout, cout, 3, 1, 1)
        self.fuse1, self.fuse2 = _DFBLK(cond_dim, cin), _DFBLK(cond_dim, cout)
        self.c_sc = nn.Conv2d(cin, cout, 1) if cin != cout else None

    def forward(self, x, y):
        x = F.interpolate(x, scale_factor=2)
        h = self.c2(self.fuse2(self.c1(self.fuse1(x, y)), y))
        return (self.c_sc(x) if self.c_sc is not None else x) + h


class DFGAN_G(nn.Module):
    def __init__(self, ngf=32, nz=100, cond_dim=256, imsize=256):
        super().__init__()
        self.ngf = ngf
        self.fc = nn.Linear(nz, ngf * 8 * 4 * 4)
        ch = [ngf * min(2 ** i, 8) for i in range(int(np.log2(imsize)) - 1)][::-1]
        self.GBlocks = nn.ModuleList(_GBlock(cond_dim + nz, a, b) for a, b in zip(ch[:-1], ch[1:]))
        self.to_rgb = nn.Sequential(nn.LeakyReLU(0.2), nn.Conv2d(ch[-1], 3, 3, 1, 1), nn.Tanh())

    def forward(self, z, c):
        out, cond = self.fc(z).view(z.size(0), 8 * self.ngf, 4, 4), torch.cat((z, c), 1)
        for blk in self.GBlocks:
            out = blk(out, cond)
        return self.to_rgb(out)


def make_dfgan(ckpt):
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)["model"]["netG"]
    G = DFGAN_G()
    G.load_state_dict({k[7:] if k.startswith("module.") else k: v for k, v in sd.items()})
    G.to(dev).eval()
    rnn = E.RNN_ENCODER(len(E.IXTOWORD), nhidden=256)
    rnn.load_state_dict(torch.load(E.DAMSM_DIR / "text_encoder200.pth", map_location="cpu", weights_only=True))
    rnn.to(dev).eval()
    return lambda ids, z: G(z, E.rnn_encode(rnn, ids))


# ----------------------------------------------------------------------------- protocol
def noise(n, trunc, rs):
    """DF-GAN lib/utils.truncated_noise (trunc x normal truncated at +-2), or plain normal noise (trunc 0)."""
    if trunc:
        from scipy.stats import truncnorm
        return torch.from_numpy(trunc * truncnorm.rvs(-2, 2, size=(n, 100), random_state=rs).astype(np.float32))
    return torch.from_numpy(rs.standard_normal((n, 100)).astype(np.float32))


def word_subset(cap, rs, n=18):
    """DF-GAN / AttnGAN dataset loader: captions over n words keep a random n-word subset, in order."""
    if len(cap) <= n:
        return list(cap)
    return [cap[j] for j in np.sort(rs.permutation(len(cap))[:n])]


def generated_features(gen, inc, times, trunc, subset, seed, bs):
    rs = np.random.RandomState(seed)
    feats, t0 = [], time.time()
    for t in range(times):
        for s in range(0, E.N_IMG, bs):
            caps = [E.TEST_CAPS[i * 10 + rs.randint(10)] for i in range(s, min(s + bs, E.N_IMG))]
            if subset:
                caps = [word_subset(c, rs) for c in caps]
            z = noise(len(caps), trunc, rs).to(dev)
            with torch.no_grad():
                x = gen(caps, z).float().clamp(-1, 1)
            feats.append(inc((x + 1) / 2))
        print(f"  pass {t + 1}/{times}  {time.time() - t0:.0f}s", flush=True)
    return torch.cat(feats).numpy()


def load_npz():
    if not NPZ.is_file():
        sys.exit(f"missing {NPZ}: get bird_val256_FIDK0.npz from DF-GAN's birds metadata zip (see the docstring)")
    f = np.load(NPZ)
    return f["mu"], f["sigma"], hashlib.md5(NPZ.read_bytes()).hexdigest()


def stats(feats):
    return feats.mean(0), np.cov(feats, rowvar=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["dfgan", "ssagan", "blm"])
    ap.add_argument("--ckpt")
    ap.add_argument("--tag")
    ap.add_argument("--real", action="store_true", help="FID of our cached real test images against the npz")
    ap.add_argument("--protocol", default="dfgan", choices=["dfgan", "galip"],
                    help="dfgan: 10 passes, noise truncated at 0.88; galip: 12 passes, normal noise")
    ap.add_argument("--times", type=int, help="override the protocol's number of passes")
    ap.add_argument("--trunc", type=float, help="override the protocol's truncation (0 = none)")
    ap.add_argument("--seed", type=int, default=100)   # DF-GAN's manual_seed
    ap.add_argument("--bs", type=int, default=32)
    a = ap.parse_args()
    if sys.platform == "win32":   # no idle sleep while this runs
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    mu_r, sig_r, npz_md5 = load_npz()
    inc = DFGANInception().to(dev).eval()
    if a.real:
        for name in ("real_test_256.npy", "real_test_256_uncropped.npy"):
            p = E.CACHE / name
            if not p.is_file():
                print(f"{name}: missing (run evaluation/prepare_real.py)")
                continue
            real = np.load(p)
            f = np.concatenate([inc(torch.from_numpy(real[s:s + 64]).permute(0, 3, 1, 2).to(dev).float() / 255).numpy()
                                for s in range(0, len(real), 64)])
            print(f"{name}: {len(f)} images, FID vs npz {E.frechet_distance(*stats(f), mu_r, sig_r):.3f}", flush=True)
        os._exit(0)
    times = a.times or {"dfgan": 10, "galip": 12}[a.protocol]
    trunc = {"dfgan": 0.88, "galip": 0.0}[a.protocol] if a.trunc is None else a.trunc
    gen = make_dfgan(a.ckpt) if a.model == "dfgan" else E.make_generator(a.model, a.ckpt)
    subset = a.model in ("dfgan", "ssagan")   # 18-word DAMSM LSTM models read captions through that loader
    t0 = time.time()
    feats = generated_features(gen, inc, times, trunc, subset, a.seed, a.bs)
    fid = float(E.frechet_distance(*stats(feats), mu_r, sig_r))
    res = {"tag": a.tag, "model": a.model, "ckpt": str(a.ckpt), "protocol": a.protocol, "fid": fid,
           "images": len(feats), "passes": times, "truncation": trunc, "seed": a.seed,
           "caption_words": "random 18-word subset" if subset else "whole caption",
           "inception": "torchvision Inception-v3 IMAGENET1K_V1, DF-GAN input scaling",
           "reference": f"{NPZ.name} (md5 {npz_md5})", "seconds": round(time.time() - t0)}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / f"{a.tag}.json", "w"), indent=1)
    print(f"{a.tag}: FID {fid:.3f} on {len(feats)} images ({a.protocol} protocol, trunc {trunc}, {times} passes)")
    os._exit(0)   # CUDA + transformers threads sometimes segfault at interpreter exit on Windows


if __name__ == "__main__":
    main()
