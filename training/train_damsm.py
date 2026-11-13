"""Step 2: pre-train the BERT-DAMSM text-image encoders on CUB.

Trains the BERT text encoder (projections + the top BERT layers) together with the DAMSM image
encoder's two heads (Inception-v3 backbone frozen) using AttnGAN's DAMSM word + sentence losses, so
text and images live in one shared 256-d space. The GAN (train_gan.py) then uses
  * the text encoder (frozen) to condition the generator, and
  * the image encoder (frozen) for its DAMSM loss — now in the same space as the text vectors.

Validation after every epoch: R-precision of real test images against their captions (top-1 of 100).

Example (Colab or Kaggle, ~1-3 min per epoch on a T4):
  python training/train_damsm.py --data_dir /content/data/birds --out_dir <run folder>/damsm
Re-running with the same --out_dir resumes from damsm_state.pth.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import CUBText, eval_caption_index, seed_worker  # noqa: E402
from image_encoder import ImageEncoder  # noqa: E402
from losses import sent_loss, words_loss  # noqa: E402
from metrics import r_precision  # noqa: E402
from text_encoder import BertTextEncoder  # noqa: E402


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--bert", default="bert-base-uncased", help="HF name or local folder")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=48)
    ap.add_argument("--lr", type=float, default=1e-3, help="projections + image heads")
    ap.add_argument("--bert_lr", type=float, default=2e-5, help="fine-tuned BERT layers")
    ap.add_argument("--finetune_layers", type=int, default=2, help="top BERT layers to fine-tune (0-12)")
    ap.add_argument("--max_len", type=int, default=40, help="max word pieces per caption")
    ap.add_argument("--gammas", type=float, nargs=3, default=[4.0, 5.0, 10.0], help="AttnGAN DAMSM gamma1-3")
    ap.add_argument("--warmup_steps", type=int, default=300)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--init_image_encoder", default="", help="optional: start from AttnGAN image_encoder200.pth")
    ap.add_argument("--val_every", type=int, default=1)
    ap.add_argument("--time_limit_hours", type=float, default=0, help="stop cleanly before a session limit (0 = off)")
    ap.add_argument("--max_steps", type=int, default=0, help="debug: stop each epoch after N steps")
    ap.add_argument("--val_images", type=int, default=0, help="debug: validate on the first N test images")
    ap.add_argument("--seed", type=int, default=100)
    return ap.parse_args()


@torch.no_grad()
def validate(text, img, test, device, bs=64, n_images=0):
    text.eval(); img.eval()
    n = n_images or len(test)
    codes = []
    for s in range(0, n, bs):
        x = torch.stack([test.image(i) for i in range(s, min(n, s + bs))]).to(device)
        codes.append(img(x)[1].float().cpu())
    table = []
    caps = test.captions[:n * 10]
    for s in range(0, len(caps), 256):
        table.append(text(caps[s:s + 256])[1].float().cpu())
    cap_idx = [eval_caption_index(i) for i in range(n)]
    text.train(); img.train()
    return r_precision(torch.cat(codes), cap_idx, torch.cat(table))


def main():
    a = parse()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    t_start = time.time()

    train = CUBText(a.data_dir, "train", imsize=256)
    test = CUBText(a.data_dir, "test", imsize=256, caption="eval")
    loader = DataLoader(train, batch_size=a.batch_size, shuffle=True, drop_last=True, num_workers=a.workers,
                        pin_memory=device.type == "cuda", worker_init_fn=seed_worker,
                        persistent_workers=a.workers > 0)
    print(f"train {len(train)} images, test {len(test)} images, {len(loader)} steps/epoch", flush=True)

    text = BertTextEncoder(a.bert, 256, a.max_len, a.finetune_layers).to(device)
    if a.init_image_encoder:
        img = ImageEncoder.load(a.init_image_encoder)
    else:
        img = ImageEncoder(256, imagenet_weights=True)
    img.to(device)
    bert_params = [p for p in text.bert.parameters() if p.requires_grad]
    head_params = list(text.proj_words.parameters()) + list(text.proj_sent.parameters()) + \
        list(img.emb_features.parameters()) + list(img.emb_cnn_code.parameters())
    groups = [{"params": head_params, "lr": a.lr}]
    if bert_params:
        groups.append({"params": bert_params, "lr": a.bert_lr})
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.999), weight_decay=0.0)
    total = a.epochs * len(loader)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup_steps) *
                                              (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total)))))

    start, best = 1, -1.0
    state_path = out / "damsm_state.pth"
    if state_path.is_file():
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        text.load_state_dict(st["text"], strict=False)
        img.load_state_dict(st["img"], strict=False)
        opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        start, best = st["epoch"] + 1, st["best"]
        print(f"resumed from epoch {st['epoch']} (best R-precision {best:.2f}%)", flush=True)

    log_path = out / "damsm_log.csv"
    new_log = not log_path.is_file()
    log = open(log_path, "a", newline="")
    w = csv.writer(log)
    if new_log:
        w.writerow(["epoch", "loss", "w_loss", "s_loss", "r_precision", "r_precision_std", "minutes"])

    trainable = [p for g in groups for p in g["params"]]
    text.train(); img.train()
    epoch_time = 0.0
    for epoch in range(start, a.epochs + 1):
        t0 = time.time()
        sums, n = np.zeros(3), 0
        for step, (x, caps, cls, _) in enumerate(loader):
            if a.max_steps and step >= a.max_steps:
                break
            x, cls = x.to(device, non_blocking=True), cls.to(device)
            with torch.no_grad():
                regions, pooled = img.features(x)
            regions, code = img.heads(regions, pooled)
            words, sent, lens = text(list(caps))
            w0, w1 = words_loss(regions, words, lens, cls, *a.gammas)
            s0, s1 = sent_loss(code, sent, cls, a.gammas[2])
            loss = w0 + w1 + s0 + s1
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step(); sched.step()
            sums += [loss.item(), (w0 + w1).item(), (s0 + s1).item()]; n += 1
            if step % 50 == 0:
                print(f"  epoch {epoch} step {step}/{len(loader)} loss {loss.item():.3f}", flush=True)
        rp = (float("nan"), 0.0)
        if epoch % a.val_every == 0 or epoch == a.epochs:
            rp = validate(text, img, test, device, n_images=a.val_images)
        m = sums / max(n, 1)
        epoch_time = time.time() - t0
        print(f"epoch {epoch}: loss {m[0]:.3f} (words {m[1]:.3f}, sent {m[2]:.3f})  "
              f"R-precision {rp[0]:.2f}% +- {rp[1]:.2f}  {epoch_time / 60:.1f} min", flush=True)
        w.writerow([epoch, *[round(v, 4) for v in m], round(rp[0], 3), round(rp[1], 3), round(epoch_time / 60, 2)])
        log.flush()
        if rp[0] == rp[0] and rp[0] > best:
            best = rp[0]
            text.save(out / "text_encoder.pth", epoch=epoch, r_precision=rp[0])
            torch.save(img.state_dict(), out / "image_encoder.pth")
            print(f"  saved best encoders (R-precision {best:.2f}%)", flush=True)
        tmp = out / "damsm_state.tmp"
        torch.save({"text": text.trainable_state_dict(),
                    "img": {k: v for k, v in img.state_dict().items() if k.startswith("emb_")},
                    "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": epoch, "best": best,
                    "args": vars(a)}, tmp)
        os.replace(tmp, state_path)
        if a.time_limit_hours and (time.time() - t_start + 1.2 * epoch_time) / 3600 > a.time_limit_hours:
            print("time limit reached - run the same command again to resume", flush=True)
            break
    log.close()
    print(f"done. best R-precision {best:.2f}%  encoders in {out}", flush=True)


if __name__ == "__main__":
    main()
