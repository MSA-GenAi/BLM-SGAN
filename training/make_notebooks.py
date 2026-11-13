"""Builds the Colab notebooks (valid nbformat 4 JSON). Edit this file, not the .ipynb files, then run

    python training/make_notebooks.py

  colab_train.ipynb         train BLM-SGAN: step 2 (BERT-DAMSM text + image encoders), step 3 (GAN), export
  colab_eval_epochs.ipynb   score every saved generator of a run (evaluation/run_epochs.py)
"""
import json
from pathlib import Path

REPO_URL = "https://github.com/MSA-GenAi/BLM-SGAN.git"
CELLS = []


def md(text):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    text = text.replace("@@REPO_URL@@", REPO_URL)
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n").splitlines(keepends=True)})


def save(path):
    nb = {"cells": CELLS, "metadata": {"accelerator": "GPU",
                                       "kernelspec": {"display_name": "Python 3", "language": "python",
                                                      "name": "python3"},
                                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    for i, c in enumerate(nb["cells"]):
        c["id"] = f"cell{i:02d}"
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("wrote", path, len(CELLS), "cells")
    CELLS.clear()


CLONE = r"""
def github_token():   # only for a private copy: the key icon in the left bar, name GITHUB_TOKEN, "Notebook access" on
    try:
        from google.colab import userdata
        return userdata.get("GITHUB_TOKEN") or ""
    except Exception:
        return ""

if os.path.isdir(CODE):
    shutil.rmtree(CODE)
tok = github_token()
url = REPO_URL.replace("https://", f"https://x-access-token:{tok}@") if tok else REPO_URL
r = subprocess.run(["git", "clone", "-q", "--depth", "1", "-b", REPO_BRANCH, url, CODE], capture_output=True, text=True)
if r.returncode != 0:
    raise RuntimeError("git clone failed:\n" + (r.stderr.replace(tok, "***") if tok else r.stderr))
subprocess.run(["git", "-C", CODE, "remote", "set-url", "origin", REPO_URL])   # never keep a token on disk
commit = subprocess.run(["git", "-C", CODE, "log", "-1", "--format=%h %s"], capture_output=True, text=True).stdout.strip()
print("code ready:", CODE, "|", commit)
"""

CUB = r"""
CUB_URL = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"
CUB_MD5 = "97eceeb196236b17998738112f37df78"

def md5(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

if not os.path.isdir(f"{DATA}/CUB_200_2011/images"):
    os.makedirs(DATA, exist_ok=True)
    !wget -q -O /tmp/CUB_200_2011.tgz "{CUB_URL}"
    assert md5("/tmp/CUB_200_2011.tgz") == CUB_MD5, "CUB archive is corrupted - delete it and rerun"
    !tar -xzf /tmp/CUB_200_2011.tgz -C {DATA} CUB_200_2011/images CUB_200_2011/images.txt CUB_200_2011/bounding_boxes.txt
    os.remove("/tmp/CUB_200_2011.tgz")
"""


# ================================================================================================= training
def colab_train():
    md(r"""
# BLM-SGAN: training on Colab

BLM-SGAN conditions SSA-GAN's generator on a BERT text encoder trained for bird captions. This notebook trains it
in two steps and exports the result for the web app:

| Section | What | Time |
|---|---|---|
| 2 | BERT-DAMSM: the BERT text encoder and a DAMSM image encoder, trained together on CUB | ~2 h on a T4 (40 epochs) |
| 3 | The GAN, 600 epochs; FID / IS / R-precision every `EVAL_EVERY` epochs | many sessions; the first log lines print the speed |
| 4 | Export the best generator + its text encoder for `checkpoints/birds/` | seconds |

The reported model was trained with these settings (batch 24 on an A100, no Fourier loss) and is epoch 555.

**Before the first run:** *Runtime → Change runtime type → A100* (or a T4 with `BATCH_SIZE = 12`). Nothing to prepare
on Google Drive: the notebook creates **My Drive/BLM-SGAN/**, takes the CUB captions from the repository and the
images from Caltech. A run needs about 15 GB of free Drive space.

**Sessions:** Colab ends sessions without warning. Both steps write their full state to Drive after every epoch, so
*Runtime → Run all* continues where they stopped.
""")

    code(r"""
import os, sys, glob, shutil, hashlib, subprocess, time, csv

# --- this run --------------------------------------------------------------------------------------
EXPERIMENT = "blm_sgan"     # folder of this run on Drive
FOURIER_WEIGHT = 0.0        # optional Fourier loss (0 = off, as in the reported model)
FOURIER_MODE = "batch"      # spectrum statistics of the generated batch vs the real batch

# --- Google Drive ------------------------------------------------------------------------------------
ROOT = "/content/drive/MyDrive/BLM-SGAN"
DAMSM_DIR = f"{ROOT}/damsm"             # step 2: the BERT-DAMSM encoders (trained once, shared by all runs)
OUT = f"{ROOT}/runs/{EXPERIMENT}"       # step 3: state.pth, samples/, metrics.csv, models/
IMAGE_ENCODER_INIT = f"{ROOT}/image_encoder200.pth"   # optional: AttnGAN's DAMSM image encoder as the starting
                                                      # point of step 2 (otherwise ImageNet weights)

# --- code --------------------------------------------------------------------------------------------
REPO_URL = "@@REPO_URL@@"
REPO_BRANCH = "main"

# --- training ----------------------------------------------------------------------------------------
DAMSM_EPOCHS = 40
GAN_EPOCHS = 600
BATCH_SIZE = 24         # needs ~22 GB (A100); 12 fits a 15-16 GB T4
SAVE_EVERY = 5          # keep models/netG_<epoch>.pth every N epochs (118 MB each), for the epoch selection
EVAL_EVERY = 5          # FID / IS / R-precision on the 2,933 test images -> metrics.csv, models/netG_best.pth
TIME_LIMIT_HOURS = 0    # 0 = until Colab ends the session (the state is saved every epoch)

DATA = "/content/data/birds"   # local disk (fast), rebuilt every session
CODE = "/content/code"
NB_START = time.time()

def hours_left():
    return TIME_LIMIT_HOURS - (time.time() - NB_START) / 3600

def damsm_epochs_done():
    try:
        rows = list(csv.reader(open(f"{DAMSM_DIR}/damsm_log.csv")))[1:]
        return max(int(r[0]) for r in rows if r)
    except (OSError, ValueError):
        return 0

def encoder_ready():   # the GAN must use the finished encoder, never a half-trained one
    return (damsm_epochs_done() >= DAMSM_EPOCHS and os.path.isfile(f"{DAMSM_DIR}/text_encoder.pth")
            and os.path.isfile(f"{DAMSM_DIR}/image_encoder.pth"))
""")

    code(r"""
from google.colab import drive
drive.mount("/content/drive")
import torch
assert torch.cuda.is_available(), "No GPU: Runtime -> Change runtime type -> GPU, then run again"
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
GPU_GB = torch.cuda.get_device_properties(0).total_memory / 2**30
if 2.3 + 0.8 * BATCH_SIZE > 0.95 * GPU_GB:   # measured: ~2.3 GB + ~0.8 GB per image
    print(f"WARNING: batch {BATCH_SIZE} probably does not fit in {GPU_GB:.1f} GB")
os.makedirs(DAMSM_DIR, exist_ok=True)
os.makedirs(OUT, exist_ok=True)
print(f"run {EXPERIMENT}: batch {BATCH_SIZE}, {GAN_EPOCHS} epochs, Fourier weight {FOURIER_WEIGHT}")
print("run folder:", OUT)
print("text encoder:", "ready" if encoder_ready() else f"{damsm_epochs_done()}/{DAMSM_EPOCHS} epochs - section 2 trains it")
""")

    md("## 1. Code and data")
    code(CLONE)
    code(CUB + r"""
META = f"{CODE}/training/cub_meta/birds"   # captions + train/test splits (AttnGAN's preprocessed CUB metadata)
for f in ["captions.pickle", "train/filenames.pickle", "train/class_info.pickle",
          "test/filenames.pickle", "test/class_info.pickle"]:
    os.makedirs(os.path.dirname(f"{DATA}/{f}"), exist_ok=True)
    shutil.copy(f"{META}/{f}", f"{DATA}/{f}")
n = len(glob.glob(f"{DATA}/CUB_200_2011/images/*/*.jpg"))
assert n == 11788, n
print(n, "images ready in", DATA)
""")

    md(r"""
## 2. BERT-DAMSM text and image encoders (once)

Skipped when `DAMSM_DIR` already holds a finished encoder. Validation prints the R-precision of **real** test images
against their captions (top-1 of 100, chance 1%): the encoder of the reported model reached 16.3%, and AttnGAN's
LSTM-DAMSM scores 16.9% on the same test.
""")
    code(r"""
if encoder_ready():
    print(f"encoder ready ({damsm_epochs_done()} epochs) - nothing to train")
else:
    use_init = bool(IMAGE_ENCODER_INIT) and os.path.isfile(IMAGE_ENCODER_INIT)
    print("image encoder start:", IMAGE_ENCODER_INIT if use_init else "ImageNet weights")
    init = f"--init_image_encoder '{IMAGE_ENCODER_INIT}'" if use_init else ""
    limit = f"{hours_left():.2f}" if TIME_LIMIT_HOURS else "0"
    !cd {CODE} && python -u training/train_damsm.py --data_dir {DATA} --out_dir "{DAMSM_DIR}" --epochs {DAMSM_EPOCHS} --time_limit_hours {limit} {init}
    if globals().get("_exit_code", 0) != 0:   # set by IPython after a ! command
        raise RuntimeError("train_damsm.py failed - see the messages above")
!tail -n 3 "{DAMSM_DIR}/damsm_log.csv"
""")

    md(r"""
## 3. GAN training

Every epoch the full state goes to `state.pth` in the run folder; every 5 epochs a grid `samples/epoch_XXX.png`
(8 test captions × 4 noise vectors, captions in `samples/captions.txt`). Every `EVAL_EVERY` epochs: FID, IS and
R-precision in `metrics.csv`; the best-FID generator is `models/netG_best.pth`. The first epoch's step lines print the
speed (s/step, estimated epoch time) and the peak GPU memory.
""")
    code(r"""
if not encoder_ready():
    raise RuntimeError(f"the text encoder is not finished ({damsm_epochs_done()}/{DAMSM_EPOCHS} epochs) - run section 2")
flags = (f"--fourier_weight {FOURIER_WEIGHT} --fourier_mode {FOURIER_MODE} --save_every {SAVE_EVERY} "
         f"--eval_every {EVAL_EVERY}")
limit = f"{hours_left():.2f}" if TIME_LIMIT_HOURS else "0"
!cd {CODE} && python -u training/train_gan.py --data_dir {DATA} --damsm_dir "{DAMSM_DIR}" --out_dir "{OUT}" --epochs {GAN_EPOCHS} --batch_size {BATCH_SIZE} --gpus 1 --time_limit_hours {limit} {flags}
if globals().get("_exit_code", 0) != 0:   # everything up to the last finished epoch is on Drive
    raise RuntimeError("train_gan.py failed - see the messages above")
""")
    code(r"""
# progress (also works on its own after a restart: run the first cell, then this one)
import pandas as pd
from IPython.display import Image, display
def table(name):
    try:
        return pd.read_csv(f"{OUT}/{name}")
    except (OSError, pd.errors.EmptyDataError):
        return pd.DataFrame()
log, met = table("train_log.csv"), table("metrics.csv")
print(log.tail(3).to_string(index=False) if len(log) else "no finished GAN epoch yet")
if len(met):
    print(met.to_string(index=False))
    if len(met) > 1:
        met.plot(x="epoch", y=["fid", "r_precision"], subplots=True, figsize=(8, 5), marker="o")
samples = sorted(glob.glob(f"{OUT}/samples/epoch_*.png"))
if samples:
    print(open(f"{OUT}/samples/captions.txt").read()); display(Image(samples[-1]))
""")

    md(r"""
## 4. Export for the web app

Writes one zip with the best generator (by FID during training) and its text encoder to the run folder on Drive.
Unzip it into `checkpoints/birds/` of the repository and restart the app. To choose the epoch more carefully, score
every saved generator with `colab_eval_epochs.ipynb` and export that epoch with `training/export_model.py`.
""")
    code(r"""
import json, re
best = f"{OUT}/models/best.txt"
if not os.path.isfile(best):
    print("nothing to export yet: the first evaluation is at epoch", EVAL_EVERY)
else:
    epoch = int(re.search(r"epoch (\d+)", open(best).read()).group(1))
    exp = "/content/export/blm_sgan"
    os.makedirs(exp, exist_ok=True)
    shutil.copy(f"{OUT}/models/netG_best.pth", f"{exp}/netG_{epoch:03d}.pth")
    shutil.copy(f"{OUT}/models/text_encoder.pth", f"{exp}/text_encoder.pth")
    json.dump({"name": "BLM-SGAN", "kind": "blm", "order": 5,
               "description": "Our model: a BERT text encoder trained on bird captions guides an SSA-GAN generator."},
              open(f"{exp}/model.json", "w"), indent=1)
    zip_path = shutil.make_archive(f"{OUT}/blm_sgan_epoch{epoch}", "zip", "/content/export", "blm_sgan")
    print("zip on Drive:", zip_path)
""")


# ============================================================================================== evaluation
def colab_eval():
    md(r"""
# BLM-SGAN: evaluate the saved generators (Colab)

Scores every `netG_<epoch>.pth` of a run **straight from Google Drive**, with the code and protocol of `evaluation/`:

* **Stage 1, every epoch** in `FROM_EPOCH … TO_EPOCH`: 2,933 images (one per test image) → FID against the whole
  test photos (SSA-GAN's protocol), FID against the bird crops, KID, FD_CLIP, precision / recall / density /
  coverage, CLIP score, R-precision (CLIP and DAMSM), Inception Scores.
* **Stage 2, the `X10_TOP` best epochs** by SSA-GAN's FID: 29,330 images (10 captions per test image) → the CUB
  Inception Score of the CUB papers and the 30k FID.

On this protocol SSA-GAN's released model scores FID 15.48 (paper: 15.61) and IS 5.10 ± 0.06 (paper: 5.17 ± 0.08).

**Before the first run:** *Runtime → Change runtime type → GPU* (an A100 is fastest), and upload **`eval_assets.zip`**
(StackGAN's CUB Inception converted by `evaluation/inception_cub.py` + AttnGAN's DAMSM encoders) to `MyDrive/BLM-SGAN/`.

Results go to `runs/<EXPERIMENT>/eval/` on Drive after every epoch, so *Run all* after a disconnect continues where it
stopped. At the end, download **`runs/<EXPERIMENT>/eval_results.zip`**.
""")

    code(r"""
import os, sys, glob, shutil, hashlib, subprocess, time, json, zipfile

# --- what to evaluate --------------------------------------------------------------------------------
EXPERIMENT = "blm_sgan"            # the run folder on Drive (runs/<EXPERIMENT>/models/netG_<epoch>.pth)
FROM_EPOCH, TO_EPOCH = 355, 600    # the last 50 saved generators (one every 5 epochs)
X10_TOP = 10                       # stage 2 for the 10 best epochs by SSA-GAN's FID (0 = off)
PREFIX = "blm_sgan"                # result names: <PREFIX>_<epoch>.json
BATCH = 64                         # images per batch (the results do not depend on it)

# --- Google Drive ------------------------------------------------------------------------------------
ROOT = "/content/drive/MyDrive/BLM-SGAN"
RUN = f"{ROOT}/runs/{EXPERIMENT}"
MODELS = f"{RUN}/models"
ASSETS = f"{ROOT}/eval_assets.zip"
EVAL_RESULTS = f"{RUN}/eval"             # JSON results: on Drive, survive the session
EVAL_CACHE = "/content/eval_cache"       # generated images + features: local disk

# --- code --------------------------------------------------------------------------------------------
REPO_URL = "@@REPO_URL@@"
REPO_BRANCH = "main"
DATA = "/content/data/birds"
CODE = "/content/code"

ASSET_MD5 = {"pretrained/inception_cub.pth": "24907d3dff08ea2c2c8e3a4425c0e998",
             "DAMSMencoders/bird/text_encoder200.pth": "11fe9392efb4f231fd29f2cf77ff357e",
             "DAMSMencoders/bird/image_encoder200.pth": "c590e9c700694ea7cd4fcd2b01f97ef2"}
""")

    code(r"""
from google.colab import drive
drive.mount("/content/drive")
import re, torch
assert torch.cuda.is_available(), "No GPU: Runtime -> Change runtime type -> GPU, then run again"
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
ckpts = sorted(int(m.group(1)) for p in glob.glob(f"{MODELS}/netG_*.pth")
               if (m := re.fullmatch(r"netG_(\d+)\.pth", os.path.basename(p))) and FROM_EPOCH <= int(m.group(1)) <= TO_EPOCH)
assert ckpts, f"no netG_<epoch>.pth between {FROM_EPOCH} and {TO_EPOCH} in {MODELS}"
assert os.path.isfile(f"{MODELS}/text_encoder.pth"), f"{MODELS}/text_encoder.pth is missing"
print(f"{len(ckpts)} generators: epochs {ckpts[0]} ... {ckpts[-1]}")
if not os.path.isfile(ASSETS):
    raise RuntimeError(f"Upload eval_assets.zip to {ROOT}/ first (see the top of the notebook)")
os.makedirs(EVAL_RESULTS, exist_ok=True)
print(f"results folder: {EVAL_RESULTS} ({len(glob.glob(f'{EVAL_RESULTS}/{PREFIX}_*.json'))} epochs already done)")
""")

    md("## 1. Code, packages and data")
    code("!pip -q install pytorch-fid\n" + CLONE)
    code(CUB + r"""
# CUB captions + test split -> where evaluation/ expects them (data/birds of the repository)
META = f"{CODE}/training/cub_meta/birds"
for f in ["captions.pickle", "test/filenames.pickle", "test/class_info.pickle"]:
    os.makedirs(os.path.dirname(f"{CODE}/data/birds/{f}"), exist_ok=True)
    shutil.copy(f"{META}/{f}", f"{CODE}/data/birds/{f}")
print(len(glob.glob(f"{DATA}/CUB_200_2011/images/*/*.jpg")), "CUB photos")

# CUB Inception (IS) + AttnGAN DAMSM encoders (DAMSM R-precision)
with zipfile.ZipFile(ASSETS) as z:
    z.extractall(CODE)
for f, h in ASSET_MD5.items():
    assert md5(f"{CODE}/{f}") == h, f"{f} in eval_assets.zip is corrupted - upload eval_assets.zip again"
print("eval_assets.zip: all files match")

# real-image references: test photos cropped like the training images, and uncropped (SSA-GAN's FID)
os.environ["EVAL_CACHE"], os.environ["EVAL_RESULTS"] = EVAL_CACHE, EVAL_RESULTS
os.makedirs(EVAL_CACHE, exist_ok=True)
if not os.path.isfile(f"{EVAL_CACHE}/real_test_256_uncropped.npy"):
    !cd {CODE} && python evaluation/prepare_real.py --cub {DATA}/CUB_200_2011
""")

    md(r"""
## 2. Evaluate

One line per epoch. Stage 1 takes a minute or two per epoch on an A100, stage 2 several minutes per epoch (29,330
images). The generated images are deleted after scoring; the results stay.
""")
    code(r"""
!cd {CODE} && python -u evaluation/run_epochs.py --models "{MODELS}" --prefix {PREFIX} --from_epoch {FROM_EPOCH} --to_epoch {TO_EPOCH} --x10_top {X10_TOP} --bs {BATCH} --delete_images
""")
    code(r"""
!cd {CODE} && python evaluation/summarize_epochs.py {PREFIX} --sort fid_uncropped
zip_path = shutil.make_archive(f"{RUN}/eval_results", "zip", EVAL_RESULTS)
print("\ndownload this file from Drive:", zip_path)
""")


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    assert "@@ORG@@" not in REPO_URL, "set REPO_URL to the GitHub repository first"
    colab_train()
    save(here / "colab_train.ipynb")
    colab_eval()
    save(here / "colab_eval_epochs.ipynb")
