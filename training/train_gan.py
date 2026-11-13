"""Step 3: train BLM-SGAN (SSA-GAN generator / discriminator from model.py + the BERT-DAMSM text encoder).

Losses as SSA-GAN: hinge loss with a mismatched-caption term, MA-GP (p=6, k=2) and the DAMSM loss x0.05, computed
with the image encoder trained together with the text encoder in step 2; an optional Fourier loss. Adam 1e-4 /
4e-4, betas (0, 0.9). The text encoder is frozen.
  * The full training state (G, D, both optimisers, epoch) is saved every epoch and resumed automatically.
  * FID / IS / R-precision on the test split every --eval_every epochs; netG_best.pth is kept by FID.
  * Memory: ~2.3 GB + ~0.8 GB per image, so batch 12 fits a 15-16 GB GPU and batch 24 needs ~22 GB.
The reported model: batch 24, 600 epochs, no Fourier loss, epoch 555.

Example (Colab, A100):
  python training/train_gan.py --data_dir /content/data/birds --damsm_dir <run>/damsm --out_dir <run>/gan          --batch_size 24 --time_limit_hours 0
Re-run the same command (with the out_dir restored) to continue.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import sys
import time
from pathlib import Path

if sys.platform.startswith("linux"):   # less fragmentation (Kaggle/Colab); must be set before CUDA starts
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parent))                      # repository root: model.py, sync_batchnorm
from data import CUBText, eval_caption_index, seed_worker  # noqa: E402
from image_encoder import ImageEncoder  # noqa: E402
from losses import damsm_loss, fourier_loss, ma_gp  # noqa: E402
from metrics import Inception, frechet_distance, inception_score, r_precision, stats  # noqa: E402
from text_encoder import BertTextEncoder  # noqa: E402
from model import NetD, NetG  # noqa: E402
from sync_batchnorm import DataParallelWithCallback  # noqa: E402


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--damsm_dir", required=True, help="folder with text_encoder.pth + image_encoder.pth (step 2)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--bert", default="", help="override the BERT location stored in text_encoder.pth")
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--batch_size", type=int, default=12)
    ap.add_argument("--gpus", type=int, default=1,
                    help="GPUs for the GAN. >1 = DataParallel (experimental: crashed once on Kaggle 2x T4 with "
                         "'CUDA error: unspecified launch failure')")
    ap.add_argument("--lr_g", type=float, default=1e-4)
    ap.add_argument("--lr_d", type=float, default=4e-4)
    ap.add_argument("--nf", type=int, default=64)
    ap.add_argument("--damsm_weight", type=float, default=0.05)
    ap.add_argument("--gammas", type=float, nargs=3, default=[5.0, 5.0, 10.0], help="DAMSM gamma1-3 (SSA-GAN values)")
    ap.add_argument("--fourier_weight", type=float, default=0.0, help="Fourier loss weight (0 = off, as in the reported model)")
    ap.add_argument("--fourier_mode", default="batch", choices=["batch", "paired"],
                    help="batch: fake vs real spectrum statistics (default); paired: per-image spectra against the real photo of the same caption")
    ap.add_argument("--fourier_cutoff", type=float, default=0.25, help="cycles/pixel, 0..0.5")
    ap.add_argument("--init_g", default="", help="warm start: generator weights (e.g. SSA-GAN netG_550.pth)")
    ap.add_argument("--init_d", default="", help="warm start: discriminator weights (e.g. SSA-GAN netD_550.pth)")
    ap.add_argument("--save_every", type=int, default=50, help="keep models/netG_<epoch>.pth every N epochs (118 MB each)")
    ap.add_argument("--copy_models_to", default="",
                    help="also copy every saved generator, netG_best.pth, best.txt, metrics.csv and the text encoder "
                         "to this folder (e.g. a second Google Drive folder as a backup)")
    ap.add_argument("--sample_every", type=int, default=5, help="samples/epoch_<epoch>.png every N epochs")
    ap.add_argument("--eval_every", type=int, default=20, help="FID/IS/R-precision every N epochs (0 = off)")
    ap.add_argument("--eval_images", type=int, default=0, help="test images to evaluate (0 = all 2,933)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--time_limit_hours", type=float, default=0, help="stop cleanly before a session limit (0 = off)")
    ap.add_argument("--max_steps", type=int, default=0, help="debug: stop each epoch after N steps")
    ap.add_argument("--seed", type=int, default=100)
    return ap.parse_args()


def unwrap(m: nn.Module) -> nn.Module:
    return m.module if isinstance(m, nn.DataParallel) else m


def load_weights(module: nn.Module, path: str, key: str) -> None:
    """Plain state dict (netG_*.pth / netD_*.pth, e.g. SSA-GAN's release) or a state.pth from this script."""
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(sd.get(key), dict):
        sd = sd[key]
    module.load_state_dict({(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}, strict=True)


@torch.no_grad()
def evaluate(G, text, img_enc, incep, test, real, table, device, n, bs=16):
    """Generate one image per test image (fixed caption + fixed noise) -> FID, IS, BERT-DAMSM R-precision."""
    G.eval()
    zgen = torch.Generator().manual_seed(2024)
    cap_idx = [eval_caption_index(i) for i in range(n)]
    feats, codes = [], []
    for s in range(0, n, bs):
        idx = cap_idx[s:s + bs]
        z = torch.randn(len(idx), 100, generator=zgen).to(device)
        x = G(z, text([test.captions[c] for c in idx])[1])[0]
        feats.append(incep.pool3(x))
        codes.append(img_enc(x)[1].float().cpu())
    G.train()
    fe = torch.cat(feats)
    mu, sig = stats(fe)
    return {"fid": frechet_distance(mu, sig, real["mu"], real["sigma"]),
            "is": inception_score(fe, incep.W),
            "r_precision": r_precision(torch.cat(codes), cap_idx, table)}


@torch.no_grad()
def eval_references(text, incep, test, device, n, cache: Path):
    """Real-image Inception statistics + the caption table (sentence vectors of all test captions)."""
    if cache.is_file():
        ref = torch.load(cache, weights_only=False)
        if ref["n"] == n:
            return ref
    feats = []
    for s in range(0, n, 32):
        feats.append(incep.pool3(torch.stack([test.image(i) for i in range(s, min(n, s + 32))])))
    mu, sig = stats(torch.cat(feats))
    table = torch.cat([text(test.captions[s:s + 256])[1].float().cpu() for s in range(0, len(test.captions), 256)])
    ref = {"n": n, "mu": mu, "sigma": sig, "table": table}
    torch.save(ref, cache)
    return ref


def main():
    a = parse()
    out = Path(a.out_dir)
    (out / "models").mkdir(parents=True, exist_ok=True)
    (out / "samples").mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    t_start = time.time()

    # ---------------------------------------------------------------- data
    train = CUBText(a.data_dir, "train", imsize=256)
    test = CUBText(a.data_dir, "test", imsize=256, augment=False, caption="eval")
    loader = DataLoader(train, batch_size=a.batch_size, shuffle=True, drop_last=True, num_workers=a.workers,
                        pin_memory=device.type == "cuda", worker_init_fn=seed_worker,
                        persistent_workers=a.workers > 0)
    print(f"train {len(train)} images, {len(loader)} steps/epoch, batch {a.batch_size}", flush=True)

    # ------------------------------------------------------------- encoders
    damsm = Path(a.damsm_dir)
    text = BertTextEncoder.load(damsm / "text_encoder.pth", bert=a.bert or None).to(device).eval()
    img_enc = ImageEncoder.load(damsm / "image_encoder.pth").to(device).eval()
    for p in list(text.parameters()) + list(img_enc.parameters()):
        p.requires_grad = False
    shutil.copy2(damsm / "text_encoder.pth", out / "models" / "text_encoder.pth")   # the model folder is self-contained

    def backup(*files):   # --copy_models_to: a second copy that does not depend on the run folder
        if not a.copy_models_to:
            return
        dst = Path(a.copy_models_to)
        dst.mkdir(parents=True, exist_ok=True)
        for f in files:
            try:
                shutil.copy2(f, dst / Path(f).name)
            except OSError as e:   # a failed backup must not stop the training
                print(f"  backup of {Path(f).name} failed: {e}", flush=True)

    backup(out / "models" / "text_encoder.pth")

    # ------------------------------------------------------------- GAN
    G, D = NetG(a.nf, 100).to(device), NetD(a.nf).to(device)
    optG = torch.optim.Adam(G.parameters(), lr=a.lr_g, betas=(0.0, 0.9))
    optD = torch.optim.Adam(D.parameters(), lr=a.lr_d, betas=(0.0, 0.9))
    start, best_fid = 1, float("inf")
    # state_prev.pth is the epoch before: a session that ends while state.pth is being written (Colab writes to
    # Google Drive) can leave state.pth truncated, and then training continues from the previous epoch.
    state_path, prev_path = out / "state.pth", out / "state_prev.pth"
    st = None
    for cand in (state_path, prev_path):
        if cand.is_file():
            try:
                st = torch.load(cand, map_location="cpu", weights_only=False)
                break
            except Exception as e:
                print(f"could not read {cand.name} ({type(e).__name__}: {e})", flush=True)
    if st is None and (state_path.is_file() or prev_path.is_file()):
        raise RuntimeError(f"no readable state in {out}: refusing to start from scratch over an existing run")
    if st is not None:
        G.load_state_dict(st["netG"]); D.load_state_dict(st["netD"])
        optG.load_state_dict(st["optG"]); optD.load_state_dict(st["optD"])
        start, best_fid = st["epoch"] + 1, st.get("best_fid", best_fid)
        print(f"resumed from epoch {st['epoch']} ({cand.name})", flush=True)
    else:
        if a.init_g:
            load_weights(G, a.init_g, "netG"); print("warm start G from", a.init_g, flush=True)
        if a.init_d:
            load_weights(D, a.init_d, "netD"); print("warm start D from", a.init_d, flush=True)
    seed = a.seed + start
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    n_gpu = min(a.gpus, torch.cuda.device_count()) if device.type == "cuda" else 0
    # ~2.3 GB fixed (models, optimisers, BERT) + ~0.8 GB per image: 0.66 was measured at batch 1-3 on a Quadro
    # P2000, but batch 16 ran out of memory on a 14.6 GB T4 (needed > 13.7 GB), so the real slope is higher.
    if device.type == "cuda":
        have = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
        need = 2.3 + 0.8 * a.batch_size / n_gpu
        print(f"GPU: {n_gpu} x {torch.cuda.get_device_name(0)} ({have:.1f} GB each); batch {a.batch_size} needs "
              f"~{need:.0f} GB on GPU 0", flush=True)
        if need > 0.95 * have:
            fit = int((0.95 * have - 2.3) / 0.8 * n_gpu)
            print(f"WARNING: batch {a.batch_size} probably does not fit; the largest that should fit is ~{fit} "
                  f"(12 on a 15-16 GB GPU).", flush=True)
    if n_gpu > 1:
        ids = list(range(n_gpu))
        print(f"using {n_gpu} GPUs (DataParallel)", flush=True)
        G = DataParallelWithCallback(G, device_ids=ids)
        D, img_enc_p = nn.DataParallel(D, device_ids=ids), nn.DataParallel(img_enc, device_ids=ids)
    else:
        img_enc_p = img_enc
    Dm = unwrap(D)

    # ------------------------------------------------------------- monitoring
    n_eval = a.eval_images or len(test)
    incep = ref = None
    if a.eval_every:
        incep = Inception(device)
        ref = eval_references(text, incep, test, device, n_eval, out / f"eval_reference_{n_eval}.pt")
    show = [i * len(test) // 8 for i in range(8)]                       # 8 test captions, different birds
    show_caps = [test.captions[eval_caption_index(i)] for i in show]
    (out / "samples" / "captions.txt").write_text("\n".join(show_caps), encoding="utf-8")
    z_show = torch.randn(32, 100, generator=torch.Generator().manual_seed(7)).to(device)
    with torch.no_grad():
        sent_show = text(show_caps)[1].repeat(4, 1)                     # rows = noise, columns = captions

    def csv_log(name, header):   # header only in a new or empty file (a restart from epoch 1 must not repeat it)
        path = out / name
        fresh = not path.is_file() or path.stat().st_size == 0
        f = open(path, "a", newline="")
        w = csv.writer(f)
        if fresh:
            w.writerow(header); f.flush()
        return f, w

    logf, log = csv_log("train_log.csv", ["epoch", "errD", "gp", "errG", "damsm", "fourier", "minutes"])
    metf, met = csv_log("metrics.csv", ["epoch", "fid", "is", "is_std", "r_precision", "r_precision_std"])

    epoch_time = 0.0
    slowest = 0.0        # slowest epoch so far without evaluation (training + samples + checkpoints)
    eval_time = 900.0    # an evaluation's duration; a cautious guess until the first one is measured
    for epoch in range(start, a.epochs + 1):
        t0 = time.time()
        evaluated = 0.0
        G.train(); D.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        sums, n = np.zeros(5), 0
        for step, (imgs, caps, cls, _) in enumerate(loader):
            if a.max_steps and step >= a.max_steps:
                break
            imgs, cls = imgs.to(device, non_blocking=True), cls.to(device)
            bs = imgs.size(0)
            with torch.no_grad():
                words, sent, cap_lens = text(list(caps))

            # (1) discriminator: real / mismatched / fake
            real_f = D(imgs)
            errD_real = F.relu(1.0 - Dm.COND_DNET(real_f, sent)).mean()
            errD_mis = F.relu(1.0 + Dm.COND_DNET(real_f[:bs - 1], sent[1:bs])).mean()
            fake, _ = G(torch.randn(bs, 100, device=device), sent)
            errD_fake = F.relu(1.0 + Dm.COND_DNET(D(fake.detach()), sent)).mean()
            errD = errD_real + (errD_fake + errD_mis) / 2.0
            optD.zero_grad(set_to_none=True)
            errD.backward()
            optD.step()

            # (2) MA-GP
            gp = ma_gp(D, Dm.COND_DNET, imgs, sent)
            optD.zero_grad(set_to_none=True)
            gp.backward()
            optD.step()

            # (3) generator
            errG = -Dm.COND_DNET(D(fake), sent).mean()
            dam = a.damsm_weight * damsm_loss(img_enc_p, fake, words, sent, cap_lens, cls, a.gammas) \
                if a.damsm_weight > 0 else torch.zeros((), device=device)
            fl = a.fourier_weight * fourier_loss(fake, imgs, a.fourier_cutoff, mode=a.fourier_mode) \
                if a.fourier_weight > 0 else torch.zeros((), device=device)
            optG.zero_grad(set_to_none=True)
            (errG + dam + fl).backward()
            optG.step()

            sums += [errD.item(), gp.item(), errG.item(), dam.item(), fl.item()]; n += 1
            if step % 100 == 0 or (epoch == start and step in (10, 30)):   # early lines give the speed at once
                now = time.time()
                if step == 0:   # includes start-up and cuDNN autotuning, so not a speed measurement
                    speed = f"first step {now - t0:.0f} s"
                else:           # steps since the previous line
                    rate = (now - t_mark) / (step - s_mark)
                    speed = f"{rate:.2f} s/step (epoch ~{rate * len(loader) / 60:.0f} min)"
                t_mark, s_mark = now, step
                mem = f"  peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB" if device.type == "cuda" else ""
                print(f"  epoch {epoch} step {step}/{len(loader)}  D {errD.item():.3f}  G {errG.item():.3f}  "
                      f"DAMSM {dam.item():.3f}  Fourier {fl.item():.3f}  {speed}{mem}", flush=True)

        m = sums / max(n, 1)
        epoch_time = time.time() - t0
        log.writerow([epoch, *[round(v, 5) for v in m], round(epoch_time / 60, 2)]); logf.flush()
        mem = f"  peak GPU {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB" if device.type == "cuda" else ""
        print(f"epoch {epoch}/{a.epochs}: D {m[0]:.3f} GP {m[1]:.4f} G {m[2]:.3f} DAMSM {m[3]:.3f} "
              f"Fourier {m[4]:.3f}  {epoch_time / 60:.1f} min{mem}", flush=True)

        Gm = unwrap(G)
        if (a.sample_every and epoch % a.sample_every == 0) or epoch == start or epoch == a.epochs:
            Gm.eval()
            with torch.no_grad():   # in chunks of 8: a 256x256 NetG pass needs ~0.25 GB per image
                grid = torch.cat([Gm(z_show[k:k + 8], sent_show[k:k + 8])[0] for k in range(0, len(z_show), 8)])
            Gm.train()
            vutils.save_image(grid, out / "samples" / f"epoch_{epoch:03d}.png", nrow=8, normalize=True,
                              value_range=(-1, 1))

        if a.save_every and epoch % a.save_every == 0 or epoch == a.epochs:
            torch.save(Gm.state_dict(), out / "models" / f"netG_{epoch:03d}.pth")
            backup(out / "models" / f"netG_{epoch:03d}.pth")
        if a.eval_every and (epoch % a.eval_every == 0 or epoch == a.epochs):
            te = time.time()
            r = evaluate(Gm, text, img_enc, incep, test, ref, ref["table"], device, n_eval)
            evaluated = eval_time = time.time() - te
            met.writerow([epoch, round(r["fid"], 3), round(r["is"][0], 3), round(r["is"][1], 3),
                          round(r["r_precision"][0], 2), round(r["r_precision"][1], 2)]); metf.flush()
            print(f"  eval: FID {r['fid']:.2f}  IS {r['is'][0]:.2f}  R-precision {r['r_precision'][0]:.1f}%", flush=True)
            if r["fid"] < best_fid:
                best_fid = r["fid"]
                torch.save(Gm.state_dict(), out / "models" / "netG_best.pth")
                (out / "models" / "best.txt").write_text(f"epoch {epoch}  FID {r['fid']:.3f}\n")
                backup(out / "models" / "netG_best.pth", out / "models" / "best.txt")
            backup(out / "metrics.csv")

        tmp = out / "state.tmp"
        torch.save({"netG": Gm.state_dict(), "netD": Dm.state_dict(), "optG": optG.state_dict(),
                    "optD": optD.state_dict(), "epoch": epoch, "best_fid": best_fid, "args": vars(a)}, tmp)
        if state_path.is_file():
            os.replace(state_path, prev_path)
        os.replace(tmp, state_path)
        # stop if the next epoch (plus its evaluation, if one is due) might not finish inside the limit
        slowest = max(slowest, time.time() - t0 - evaluated)
        nxt = epoch + 1
        need = 1.3 * slowest + (eval_time if a.eval_every and (nxt % a.eval_every == 0 or nxt == a.epochs) else 0)
        if a.time_limit_hours and (time.time() - t_start + need) / 3600 > a.time_limit_hours:
            print("time limit reached - run the same command again to resume", flush=True)
            break
    logf.close(); metf.close()
    print("done", flush=True)


if __name__ == "__main__":
    main()
