# Training BLM-SGAN

BLM-SGAN keeps SSA-GAN's generator and discriminator (`model.py`) and conditions them on a BERT text encoder trained
for bird captions. Training has two steps:

1. **BERT-DAMSM** (`train_damsm.py`): the BERT text encoder and a DAMSM image encoder are trained together with
   AttnGAN's DAMSM word and sentence losses, so text and images share one 256-d space.
2. **GAN** (`train_gan.py`): the generator and discriminator, conditioned on the frozen text encoder. The DAMSM loss
   uses the image encoder from step 1.

The easiest way is the Colab notebook `colab_train.ipynb`, which runs both steps and resumes after Colab ends a
session.

## Files

| File | Purpose |
|---|---|
| `colab_train.ipynb` | Colab: data, step 1, step 2, export for the web app |
| `colab_eval_epochs.ipynb` | Colab: scores every saved `netG_<epoch>.pth` of a run straight from Drive (`evaluation/run_epochs.py`), to choose the epoch to report |
| `make_notebooks.py` | Generates the notebooks: edit it, not the `.ipynb` files, then run it |
| `train_damsm.py` | **Step 1.** BERT text encoder + DAMSM image encoder |
| `train_gan.py` | **Step 2.** The GAN, conditioned on the frozen step-1 text encoder |
| `text_encoder.py` | `BertTextEncoder`: caption text → BERT → word features + mean-pooled sentence vector |
| `image_encoder.py` | DAMSM image encoder (Inception-v3 backbone frozen, trained heads) |
| `losses.py` | DAMSM word/sentence losses, MA-GP, the optional Fourier loss |
| `data.py` | CUB dataset returning caption *strings* |
| `metrics.py` | FID / IS / R-precision during training |
| `export_model.py` | Copies a trained run into `checkpoints/birds/<id>/` for the app and `evaluation/run_all.sh` |
| `cub_meta/birds/` | CUB captions and train/test splits (AttnGAN's preprocessed metadata) |

## Data

`--data_dir` must contain the CUB metadata and images:

```
birds/captions.pickle
birds/train/{filenames,class_info}.pickle
birds/test/{filenames,class_info}.pickle
birds/CUB_200_2011/{images/, images.txt, bounding_boxes.txt}
```

The metadata is in `cub_meta/birds/`. The images come from
[CUB-200-2011](https://data.caltech.edu/records/65de6-vp158) (md5 `97eceeb196236b17998738112f37df78`); the notebook
downloads them.

## Run

```bash
pip install -r requirements.txt
# step 1 (40 epochs, ~2 h on a T4)
python training/train_damsm.py --data_dir data/birds --out_dir runs/damsm
# step 2 (600 epochs, resumable; --time_limit_hours stops cleanly before a session ends)
python training/train_gan.py --data_dir data/birds --damsm_dir runs/damsm --out_dir runs/gan --batch_size 24
# use it in the app and evaluate it
python training/export_model.py --run_dir runs/gan            # -> checkpoints/birds/blm_sgan/
bash evaluation/run_all.sh
```

Running either script again with the same `--out_dir` resumes it (model, optimisers, epoch).

## Outputs of `train_gan.py`

| Path | Content |
|---|---|
| `state.pth`, `state_prev.pth` | full training state of the last and the previous epoch (~1.3 GB each); resuming falls back to `state_prev.pth` if a session ended while `state.pth` was being written |
| `models/netG_XXX.pth` | generator every `--save_every` epochs (118 MB each) |
| `models/netG_best.pth`, `best.txt` | lowest FID so far |
| `models/text_encoder.pth` | copy of the step-1 text encoder (the generator needs it) |
| `samples/epoch_XXX.png` | 8 test captions × 4 noise vectors, the same each time, every `--sample_every` epochs |
| `train_log.csv` | mean losses per epoch, minutes per epoch |
| `metrics.csv` | FID, IS, R-precision (BERT-DAMSM) every `--eval_every` epochs |

## Options

| Option | Default | Note |
|---|---|---|
| `--batch_size` | 12 | Needs ~2.3 GB + ~0.8 GB per image: 12 fits a 15–16 GB GPU, 24 (the reported model, as SSA-GAN) needs ~22 GB. The script prints the estimate, the speed of the first steps and the real peak per epoch |
| `--fourier_weight` | 0 | optional frequency loss (`losses.fourier_loss`); 0 in the reported model |
| `--fourier_mode` | `batch` | `paired` compares each image's spectrum with the real photo of the same caption |
| `--damsm_weight` | 0.05 | as SSA-GAN |
| `--init_g`, `--init_d` | – | warm start from other weights, e.g. SSA-GAN's `netG_550.pth` / `netD_550.pth` |
| `--finetune_layers` (step 1) | 2 | top BERT layers trained in step 1 |

## Choosing the epoch to report

The FID measured during training (`metrics.csv`) is for comparing epochs with each other only. It moves by about 1
between neighbouring checkpoints, and the last epoch is rarely the best. To choose the epoch, run
`colab_eval_epochs.ipynb`:

1. Set `EXPERIMENT` and the epoch range.
2. Upload `eval_assets.zip` to `My Drive/BLM-SGAN/`: StackGAN's CUB Inception converted with
   `evaluation/inception_cub.py` (`pretrained/inception_cub.pth`) and AttnGAN's DAMSM encoders (`DAMSMencoders/bird/`).
3. Run all. Every epoch is scored with SSA-GAN's protocol and the extra metrics, then the best ones again on 29,330
   images.
4. Download `runs/<EXPERIMENT>/eval_results.zip`.

For the reported run (epochs 355–600) this chose epoch 555; the results are in
[`evaluation/results/epochs/`](../evaluation/results/epochs/).
