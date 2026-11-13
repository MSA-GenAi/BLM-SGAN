"""Losses of BLM-SGAN: DAMSM (AttnGAN), MA-GP (DF-GAN / SSA-GAN) and an optional Fourier (frequency) loss."""
from __future__ import annotations

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------------------- DAMSM
def func_attention(query, context, gamma1):
    """AttnGAN GlobalAttention.func_attention.
    query: B x D x Lq (words), context: B x D x H x W (image regions) -> (B x D x Lq, B x Lq x H x W)."""
    B, Lq = query.size(0), query.size(2)
    ih, iw = context.size(2), context.size(3)
    S = ih * iw
    context = context.view(B, -1, S)
    attn = torch.bmm(context.transpose(1, 2), query)                 # B x S x Lq      Eq. (7)
    attn = F.softmax(attn.reshape(B * S, Lq), dim=1)                 # over words      Eq. (8)
    attn = attn.view(B, S, Lq).transpose(1, 2).reshape(B * Lq, S)
    attn = F.softmax(attn * gamma1, dim=1).view(B, Lq, S)            # over regions    Eq. (9)
    weighted = torch.bmm(context, attn.transpose(1, 2))              # B x D x Lq
    return weighted, attn.view(B, Lq, ih, iw)


def _class_mask(class_ids: torch.Tensor) -> torch.Tensor:
    """B x B, True where j != i comes from the same class as i (not a valid negative)."""
    m = class_ids[:, None] == class_ids[None, :]
    m.fill_diagonal_(False)
    return m


def sent_loss(cnn_code, sent, class_ids, gamma3=10.0, eps=1e-8):
    """Sentence-level DAMSM loss (both directions)."""
    B = cnn_code.size(0)
    scores = cnn_code @ sent.t()
    norm = cnn_code.norm(dim=1, keepdim=True) @ sent.norm(dim=1, keepdim=True).t()
    scores = scores / norm.clamp(min=eps) * gamma3
    if class_ids is not None:
        scores = scores.masked_fill(_class_mask(class_ids), float("-inf"))
    labels = torch.arange(B, device=scores.device)
    return F.cross_entropy(scores, labels), F.cross_entropy(scores.t(), labels)


def words_loss(regions, words, cap_lens, class_ids, gamma1=5.0, gamma2=5.0, gamma3=10.0):
    """Word-level DAMSM loss (both directions). regions: B x D x 17 x 17, words: B x D x L."""
    B = regions.size(0)
    sims = []
    for i in range(B):
        n = int(cap_lens[i])
        w = words[i:i + 1, :, :n].expand(B, -1, -1)                    # caption i against every image
        ctx, _ = func_attention(w, regions, gamma1)                    # B x D x n
        row = F.cosine_similarity(w, ctx, dim=1, eps=1e-8)             # B x n
        sims.append(torch.log(torch.exp(row * gamma2).sum(1, keepdim=True)))   # Eq. (10)
    sims = torch.cat(sims, 1) * gamma3                                 # [image j, caption i]
    if class_ids is not None:
        sims = sims.masked_fill(_class_mask(class_ids), float("-inf"))
    labels = torch.arange(B, device=sims.device)
    return F.cross_entropy(sims, labels), F.cross_entropy(sims.t(), labels)


def damsm_loss(image_encoder, images, words, sent, cap_lens, class_ids, gammas=(5.0, 5.0, 10.0)):
    """Total DAMSM loss of images w.r.t. their captions (word + sentence, both directions)."""
    g1, g2, g3 = gammas
    regions, code = image_encoder(images)
    w0, w1 = words_loss(regions, words, cap_lens, class_ids, g1, g2, g3)
    s0, s1 = sent_loss(code, sent, class_ids, g3)
    return w0 + w1 + s0 + s1


# ------------------------------------------------------------------------------------------ GAN
def ma_gp(netD, cond_dnet, images, sent, p=6.0, k=2.0):
    """Matching-aware zero-centred gradient penalty (DF-GAN), on real images + their captions."""
    images = images.detach().requires_grad_(True)
    sent = sent.detach().requires_grad_(True)
    out = cond_dnet(netD(images), sent)
    g_img, g_sent = torch.autograd.grad(out.sum(), (images, sent), create_graph=True)
    grad = torch.cat([g_img.flatten(1), g_sent.flatten(1)], dim=1)
    return k * grad.norm(2, dim=1).pow(p).mean()


# -------------------------------------------------------------------------------------- Fourier
def fourier_loss(fake, real, cutoff=0.25, high_weight=2.0, low_weight=1.0, mode="paired"):
    """Frequency-balanced spectrum loss (optional; off in the reported model).

    Frequencies are in cycles/pixel (0..0.71), the FFT is orthonormal and each band (below / above `cutoff`) is
    averaged over its own bins.
      mode="batch":  L1 between the batch-mean log-magnitude spectra, i.e. the frequency statistics of
                     fake vs real images (default).
      mode="paired": L1 between the magnitude spectra of each fake image and the real photo of the same caption.
    Measured on CUB test images (paired / batch): a different real bird 0.34 / 0.04, a blurred copy
    0.22 / 0.19, a noisy copy 0.59 / 0.44, flat grey 0.44 / 0.33. The paired version rates a blurry copy
    better than a sharp, different bird, so it cannot fall below the level of a realistic image; the batch
    version separates real images from blurred or noisy ones.
    """
    H, W = fake.shape[-2:]
    fy = torch.fft.fftfreq(H, device=fake.device)
    fx = torch.fft.fftfreq(W, device=fake.device)
    radius = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    high = (radius >= cutoff).to(fake.dtype)
    low = 1.0 - high
    mf = torch.fft.fft2(fake.float(), norm="ortho").abs()
    mr = torch.fft.fft2(real.float(), norm="ortho").abs()
    if mode == "batch":
        diff = (torch.log1p(mf).mean(0) - torch.log1p(mr).mean(0)).abs()       # C x H x W
        n = diff.size(0)
    else:
        diff = (mf - mr).abs().flatten(0, 1)                                     # (B*C) x H x W
        n = diff.size(0)
    hi = (diff * high).sum() / (high.sum() * n)
    lo = (diff * low).sum() / (low.sum() * n)
    return high_weight * hi + low_weight * lo
