"""FID / IS / R-precision during training (same formulas as evaluation/evaluate.py and
evaluation/inception_score.py, so the numbers line up with the final evaluation)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class Inception:
    """pytorch-fid Inception-v3: pool3 features (FID) + the 1008-way classifier (IS)."""

    def __init__(self, device):
        from pytorch_fid.inception import FID_WEIGHTS_URL, InceptionV3
        from torch.hub import load_state_dict_from_url
        self.net = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()
        self.W = load_state_dict_from_url(FID_WEIGHTS_URL, progress=False, map_location="cpu")["fc.weight"].double()
        self.device = device

    @torch.no_grad()
    def pool3(self, x: torch.Tensor) -> torch.Tensor:
        """x in [-1, 1] -> B x 2048 (float64, cpu)."""
        x01 = (x.to(self.device).clamp(-1, 1) + 1) / 2
        return self.net(x01)[0].flatten(1).double().cpu()


def stats(feats: torch.Tensor):
    f = feats.double().numpy()
    return f.mean(0), np.cov(f, rowvar=False)


def frechet_distance(mu1, s1, mu2, s2) -> float:
    from scipy import linalg
    s1, s2 = np.asarray(s1, np.float64), np.asarray(s2, np.float64)
    diff = mu1 - mu2

    def psd_sqrt(s):
        w, v = linalg.eigh((s + s.T) / 2, driver="evr")
        return (v * np.sqrt(np.clip(w, 0, None))) @ v.T

    tr_covmean = linalg.svdvals(psd_sqrt(s1) @ psd_sqrt(s2)).sum()
    return float(diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2 * tr_covmean)


def inception_score(pool3: torch.Tensor, W: torch.Tensor, splits: int = 10):
    logp = torch.log_softmax(pool3.double() @ W.T, dim=1)
    logp = logp[torch.randperm(len(logp), generator=torch.Generator().manual_seed(0))]
    scores, n = [], len(logp)
    for k in range(splits):
        lp = logp[k * n // splits:(k + 1) * n // splits]
        log_py = torch.logsumexp(lp, 0) - np.log(len(lp))
        scores.append(float((lp.exp() * (lp - log_py)).sum(1).mean().exp()))
    return float(np.mean(scores)), float(np.std(scores))


def r_precision(img_emb: torch.Tensor, cap_idx, table: torch.Tensor, draws=10, cands=100):
    """Top-1 accuracy of the true caption among `cands` candidates (true + captions of other images).
    Identical to evaluation/evaluate.py r_precision()."""
    img_emb = F.normalize(img_emb.float().cpu(), dim=1)
    table = F.normalize(table.float().cpu(), dim=1)
    n = len(cap_idx)
    true = (img_emb * table[cap_idx]).sum(1)
    accs = []
    for d in range(draws):
        g = torch.Generator().manual_seed(1000 + d)
        other = torch.randint(0, len(table), (n, cands - 1), generator=g)
        same_img = (other // 10) == (torch.tensor(cap_idx) // 10).unsqueeze(1)
        other[same_img] = (other[same_img] + 10) % len(table)
        sims = torch.einsum("nd,nkd->nk", img_emb, table[other])
        accs.append((true > sims.max(1).values).float().mean().item())
    return float(np.mean(accs) * 100), float(np.std(accs) * 100)
