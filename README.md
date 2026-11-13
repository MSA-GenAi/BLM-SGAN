# BLM-SGAN: Bidirectional Language-Modeling for Semantic Spatial Text-to-Image Generation

BLM-SGAN draws a bird from a sentence. A BERT text encoder trained for bird captions reads the description, and a
single-stage GAN generator (SSA-GAN's, with semantic masks) turns noise into a 256×256 image, using the masks to
decide *where* the text should change the picture. It is trained on CUB-200-2011.

Compared with SSA-GAN, the model it builds on, BLM-SGAN matches its image quality (FID 15.56 vs 15.48) and
understands captions better. It keeps 47% of a caption's effect when colour and size words are replaced by rare
synonyms (SSA-GAN: 33%), and it uses the words after the 18th in long captions, which SSA-GAN cannot read.

## Results

All numbers come from this repository's evaluation code, with the same captions and noise for every model. The
models are BLM-SGAN at epoch 555 (`checkpoints/birds/blm_sgan/`) and SSA-GAN's released generator (epoch 590).

### Image quality and text alignment

| Model | FID ↓ | IS ↑ | KID ×10³ ↓ | FD_CLIP ×10² ↓ | Precision ↑ | Recall ↑ | Density ↑ | Coverage ↑ | CLIP score ↑ | R-prec. CLIP ↑ | R-prec. DAMSM ↑ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SSA-GAN, as reported in its paper | 15.61 | 5.17 ± 0.08 | – | – | – | – | – | – | – | – | – |
| SSA-GAN (released, epoch 590) | **15.48** | **5.10** ± 0.06 | **1.96** | 6.13 | **0.602** | **0.644** | **0.435** | **0.564** | **0.2906** | **15.5%** | **72.0%** |
| **BLM-SGAN** (epoch 555) | 15.56 | 4.60 ± 0.06 | 2.68 | **5.82** | 0.558 | 0.633 | 0.369 | 0.538 | 0.2905 | 13.8% | 37.2% |
| Real test photos | – | 22.26 ± 0.40 | 0.13 | 1.37 | 0.805 | 0.824 | 0.965 | 0.968 | 0.2839 | 10.5% | 16.9% |

* **FID** follows SSA-GAN's protocol: 2,933 generated images (one per test image) against the whole test photos.
  On it, SSA-GAN's released model scores 15.48 against the 15.61 of its paper.
* **IS** uses StackGAN's Inception-v3 fine-tuned on CUB, on 29,330 images, as the CUB papers do. SSA-GAN scores
  5.10 against the 5.17 of its paper.
* **Real photos.** KID, FD_CLIP and precision / recall / density / coverage compare one half of the test photos
  with the other half, which gives an upper bound. Their IS is on the 2,933 photos.
* **R-precision** is how often an image's own caption ranks first among 100. The DAMSM judge favours SSA-GAN, which
  was trained with that exact DAMSM model; the CLIP judge (ViT-B/32) is neutral to both.

On 29,330 images (10 captions per test image), FID is 13.06 for BLM-SGAN and 12.89 for SSA-GAN.

### Language tests

[`evaluation/language_tests.py`](evaluation/language_tests.py) tests the two places where SSA-GAN's word-level LSTM
is weakest. Both models get the same captions and noise; CLIP ViT-B/32 judges, and neither model was trained with
it. The ± values are standard errors over captions.

| Test | SSA-GAN | BLM-SGAN |
|---|---|---|
| Rare words: share of the caption's effect kept ↑ | 33% | **47%** |
| Rare words: CLIP score lost to the rare words ↓ | 0.0199 ± 0.0004 | **0.0157 ± 0.0004** |
| Rare words: CLIP R-precision, original → rare caption ↑ | 14.5% → 4.3% | 15.3% → **6.3%** |
| Long captions: CLIP score of the words after the 18th ↑ | 0.2274 ± 0.0007 | **0.2323 ± 0.0007** |
| Long captions: R-precision on those words (chance 1%) ↑ | 1.3% | **2.1%** |

* **Rare words.** In 2,889 test captions, 8,355 colour and size words are replaced by synonyms that SSA-GAN's 88,550
  training captions contain at most twice (red → vermilion, white → alabaster, large → sizable). SSA-GAN's
  vocabulary drops the unknown words; BERT still reads them. Each image is scored against the original caption.
  "Kept" = (rare − random caption) / (original − random caption).
* **Long captions.** 2,480 test captions are longer than 18 words (22.8 on average). SSA-GAN reads the first 18
  words; BLM-SGAN reads the whole caption. Cut to 18 words, BLM-SGAN scores 0.2268 on the tail, so its +0.0055 ±
  0.0003 gain comes from reading those words.

### On the FID scale of DF-GAN, GALIP and TIGER

These papers compute FID their own way: torchvision's ImageNet Inception-v3 with their input scaling, DF-GAN's
precomputed statistics of the test bird crops, and about 30,000 images.
[`evaluation/fid_dfgan.py`](evaluation/fid_dfgan.py) reproduces that protocol. On DF-GAN's released bird model it
gives 12.07, against the 12.10 of DF-GAN's README.

| Model | DF-GAN protocol ↓ (10 per test image, noise × 0.88) | GALIP protocol ↓ (12 per test image) | Published |
|---|---|---|---|
| DF-GAN, released model | 12.07 | – | 12.10 |
| SSA-GAN (released) | 11.69 | 11.84 | – |
| **BLM-SGAN** (epoch 555) | being measured | being measured | – |
| GALIP (CVPR 2023) | – | – | 10.08 |
| TIGER (2025) | – | – | 9.38 |

### Choosing the epoch

The run saved a generator every 5 epochs. All 50 from epoch 355 to 600 were scored with the protocol above, and the
10 best again on 29,330 images ([`evaluation/results/epochs/`](evaluation/results/epochs/)). Epoch 555 has the
lowest FID on all four variants (2,933 or 29,330 images, whole photos or bird crops) and the best KID and coverage.
FID moves by about 1 between neighbouring checkpoints (555: 15.56, 560: 16.52, 600: 17.26), so the epoch has to be
chosen by evaluation; being the lowest of 50, epoch 555's value is slightly optimistic.

## How it works

| Stage | What |
|---|---|
| Text | BERT-base (uncased) reads the caption, up to 40 word pieces; the top 2 layers are fine-tuned. The sentence vector is the mean of the token states, projected to 256-d. |
| Text–image space | The projections are trained together with a DAMSM image encoder (AttnGAN's word and sentence losses). On real test photos the text encoder ranks the right caption first among 100 in 16.3% of cases, on par with AttnGAN's LSTM encoder (16.9%). |
| Generator | SSA-GAN's generator: noise → 4×4 features → 7 upsampling blocks to 256×256. In each block a predicted mask decides where the text-conditioned affine transform acts. |
| Training | Hinge loss with a mismatched-caption term, MA-GP (k = 2, p = 6), DAMSM loss × 0.05. Adam (1e-4 / 4e-4, β = (0, 0.9)), batch 24, 600 epochs, text encoder frozen. |

Training instructions: [`training/README.md`](training/README.md).

## Web app

`app/` is BLM-SGAN Studio: generate images from a description, see the semantic masks, compare captions with the
same noise, or compare BLM-SGAN with SSA-GAN side by side. It also has a How-it-works page with the evaluation
tables.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126   # or the CPU build
pip install -r requirements.txt
python -m app.server            # then open http://127.0.0.1:8000   (Windows: double-click start_app.bat)
```

Model files:

* **BLM-SGAN:** download `netG_555.pth` and `text_encoder.pth` from the
  [latest release](https://github.com/MSA-GenAi/BLM-SGAN/releases/latest) into `checkpoints/birds/blm_sgan/`.
* **SSA-GAN** (optional): `netG_590.pth` from [SSA-GAN's repository](https://github.com/wtliao/text2image) into
  `checkpoints/birds/ssagan/`, AttnGAN's DAMSM text encoder for CUB into `DAMSMencoders/bird/`, and
  `training/cub_meta/birds/captions.pickle` into `data/birds/`.

The public site runs the same app in public mode (`BLM_PUBLIC=1`: no How-it-works page, nothing saved). The
deployment files are in [`deploy/`](deploy/).

## Evaluation

```bash
python evaluation/prepare_real.py --cub <CUB_200_2011.tgz or the CUB_200_2011 folder>   # real-image references
bash evaluation/run_all.sh            # every model in checkpoints/birds/: FID, R-precision, KID, ..., then IS
python evaluation/language_tests.py --blm checkpoints/birds/blm_sgan/netG_555.pth --tag blm_sgan_555
python evaluation/fid_dfgan.py --model blm --ckpt checkpoints/birds/blm_sgan/netG_555.pth --tag blm_sgan_555_dfgan
EVAL_CAPS_PER_IMAGE=10 python evaluation/evaluate.py --model blm --ckpt <netG.pth> --tag <name>_x10   # 29,330 images
```

Needs `data/birds/{captions.pickle, test/filenames.pickle, test/class_info.pickle}` (from `training/cub_meta/birds/`)
and AttnGAN's `DAMSMencoders/bird/{text,image}_encoder200.pth`. The CUB-paper IS needs StackGAN's CUB Inception,
converted once with `python evaluation/inception_cub.py convert <birds_valid299/model.ckpt> pretrained/inception_cub.pth`.
`fid_dfgan.py` needs DF-GAN's `bird_val256_FIDK0.npz` in `data/birds/npz/`.

## Repository

| Path | Content |
|---|---|
| `model.py`, `DAMSM.py`, `miscc/`, `sync_batchnorm/` | SSA-GAN's generator and discriminator, AttnGAN's DAMSM encoders |
| `training/` | BERT-DAMSM pre-training, GAN training, Colab notebooks |
| `evaluation/` | evaluation scripts and every result in `evaluation/results/` |
| `app/` | the web app |
| `checkpoints/birds/` | one folder per model (`model.json`; the weights are downloaded) |
| `deploy/` | the public site: `build_bundle.py`, `modal_app.py`, a Dockerfile |

## Credits

* SSA-GAN: Liao et al., "Text to Image Generation with Semantic-Spatial Aware GAN", CVPR 2022
  ([paper](https://arxiv.org/abs/2104.00567), [code](https://github.com/wtliao/text2image)): the generator,
  the discriminator and the released baseline weights.
* AttnGAN: Xu et al., CVPR 2018 ([code](https://github.com/taoxugit/AttnGAN), MIT): the DAMSM losses and encoders,
  and the preprocessed CUB captions.
* BERT: Devlin et al., 2019 ([bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased)).
* DF-GAN ([code](https://github.com/tobran/DF-GAN)): the FID protocol and reference statistics of the later CUB
  papers. StackGAN's CUB Inception model for the Inception Score.
* CUB-200-2011 (Wah et al., 2011) with the captions of Reed et al., 2016.
