"""CUB-200-2011 birds with their 10 captions per image, as caption *strings* for BERT.

data_dir layout (AttnGAN's birds metadata + the CUB images):
  data_dir/captions.pickle                    [train_ids, test_ids, ixtoword, wordtoix]
  data_dir/{train,test}/filenames.pickle
  data_dir/{train,test}/class_info.pickle
  data_dir/CUB_200_2011/{images/, images.txt, bounding_boxes.txt}

Captions are rebuilt from captions.pickle (lower-case words joined by spaces), i.e. the tokenised text of
AttnGAN's preprocessed CUB captions, so no extra text files are needed.
"""
from __future__ import annotations

import pickle
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

CAPS_PER_IMAGE = 10


def eval_caption_index(i: int, k: int = 0) -> int:
    """Caption used for test image i in evaluation (same rule as evaluation/evaluate.py)."""
    return i * CAPS_PER_IMAGE + (i * 7 + k * 3) % CAPS_PER_IMAGE


def load_bboxes(cub_dir: Path) -> dict:
    id2file = dict(l.split() for l in (cub_dir / "images.txt").read_text().strip().splitlines())
    out = {}
    for line in (cub_dir / "bounding_boxes.txt").read_text().strip().splitlines():
        i, x, y, w, h = line.split()
        out[id2file[i][:-4]] = [int(float(v)) for v in (x, y, w, h)]
    return out


def crop_bird(img: Image.Image, b) -> Image.Image:
    """Same crop as datasets.get_imgs(): square of 1.5 x the longer bbox side around its centre."""
    W, H = img.size
    r = int(np.maximum(b[2], b[3]) * 0.75)
    cx, cy = int((2 * b[0] + b[2]) / 2), int((2 * b[1] + b[3]) / 2)
    return img.crop([max(0, cx - r), max(0, cy - r), min(W, cx + r), min(H, cy + r)])


class CUBText(Dataset):
    def __init__(self, data_dir, split="train", imsize=256, augment=None, caption="random"):
        """augment: random crop + flip (default: on for train). caption: "random" (one of the 10 at
        random, for training) or "eval" (the fixed evaluation caption)."""
        self.dir = Path(data_dir)
        self.split, self.imsize, self.caption = split, imsize, caption
        with open(self.dir / "captions.pickle", "rb") as f:
            meta = pickle.load(f)
        ixtoword = meta[2]
        ids = meta[0] if split == "train" else meta[1]
        self.captions = [" ".join(ixtoword[j] for j in c) for c in ids]
        with open(self.dir / split / "filenames.pickle", "rb") as f:
            self.filenames = pickle.load(f)
        with open(self.dir / split / "class_info.pickle", "rb") as f:
            self.class_ids = [int(c) for c in pickle.load(f, encoding="bytes")]
        assert len(self.captions) == CAPS_PER_IMAGE * len(self.filenames), "captions.pickle does not match filenames"
        self.bbox = load_bboxes(self.dir / "CUB_200_2011")
        augment = (split == "train") if augment is None else augment
        norm = [T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        if augment:
            self.transform = T.Compose([T.Resize(int(imsize * 76 / 64)), T.RandomCrop(imsize),
                                        T.RandomHorizontalFlip(), *norm])
        else:
            self.transform = T.Compose([T.Resize((imsize, imsize), interpolation=T.InterpolationMode.BICUBIC), *norm])

    def __len__(self):
        return len(self.filenames)

    def image(self, i: int) -> torch.Tensor:
        key = self.filenames[i]
        img = Image.open(self.dir / "CUB_200_2011" / "images" / f"{key}.jpg").convert("RGB")
        return self.transform(crop_bird(img, self.bbox[key]))

    def caption_index(self, i: int) -> int:
        if self.caption == "eval":
            return eval_caption_index(i)
        return i * CAPS_PER_IMAGE + random.randrange(CAPS_PER_IMAGE)

    def __getitem__(self, i):
        c = self.caption_index(i)
        return self.image(i), self.captions[c], self.class_ids[i], c


def seed_worker(worker_id: int) -> None:
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)
