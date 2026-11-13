"""Copy a trained model into checkpoints/birds/<id>/ so the web app and evaluation/run_all.sh use it.

  python training/export_model.py --run_dir <train_gan.py out_dir> [--epoch best|<n>] [--id blm_sgan]

The folder gets netG_<epoch>.pth, text_encoder.pth and model.json {"kind": "blm"}.
"""
import argparse
import json
import re
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--epoch", default="best", help="'best' (lowest FID during training) or an epoch number")
    ap.add_argument("--id", default="blm_sgan")
    ap.add_argument("--name", default="BLM-SGAN")
    ap.add_argument("--description", default="BERT-DAMSM text encoder, SSA-GAN generator")
    ap.add_argument("--order", type=int, default=5)
    a = ap.parse_args()
    models = Path(a.run_dir) / "models"
    if a.epoch == "best":
        src = models / "netG_best.pth"
        m = re.search(r"epoch (\d+)", (models / "best.txt").read_text())
        epoch = int(m.group(1))
    else:
        epoch = int(a.epoch)
        src = models / f"netG_{epoch:03d}.pth"
    dst = REPO / "checkpoints" / "birds" / a.id
    dst.mkdir(parents=True, exist_ok=True)
    for old in dst.glob("netG_*.pth"):
        old.unlink()
    shutil.copy2(src, dst / f"netG_{epoch:03d}.pth")
    shutil.copy2(models / "text_encoder.pth", dst / "text_encoder.pth")
    (dst / "model.json").write_text(json.dumps({
        "name": a.name, "kind": "blm", "order": a.order,
        "description": f"{a.description}, epoch {epoch}",
        "source": str(Path(a.run_dir).resolve())}, indent=1), encoding="utf-8")
    print(f"exported epoch {epoch} -> {dst}")


if __name__ == "__main__":
    main()
