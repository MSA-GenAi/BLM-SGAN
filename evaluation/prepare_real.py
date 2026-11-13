"""The 2,933 CUB test images as FID references, in two versions.

  real_test_256.npy            cropped around the bird bounding box exactly like datasets.py get_imgs() (what the
                               generators are trained on; the main FID reference)
  real_test_256_uncropped.npy  the whole photo resized to 256x256 (the reference behind SSA-GAN's reported FID:
                               SSA-GAN 590 scores 15.48 against it with one image per test image, paper 15.61)

Usage: python evaluation/prepare_real.py --cub <CUB_200_2011.tgz, or the extracted CUB_200_2011 folder>
Output: evaluation/cache/real_test_256*.npy  (N x 256 x 256 x 3 uint8, order = data/birds/test/filenames.pickle)
"""
import argparse
import io
import os
import pickle
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("EVAL_CACHE", Path(__file__).resolve().parent / "cache"))


def read_cub(path, names):
    path = Path(path)
    if path.is_dir():
        root = path if (path / "images.txt").is_file() else path / "CUB_200_2011"
        get = lambda rel: (root / rel).read_bytes()  # noqa: E731
        return [get(f"images/{k}.jpg") for k in names], get("bounding_boxes.txt").decode(), get("images.txt").decode()
    want = {f"CUB_200_2011/images/{k}.jpg": i for i, k in enumerate(names)}
    imgs, bbox_txt, images_txt = [None] * len(names), None, None
    with tarfile.open(path, "r:gz") as tf:
        for m in tf:
            if m.name in want:
                imgs[want[m.name]] = tf.extractfile(m).read()
            elif m.name == "CUB_200_2011/bounding_boxes.txt":
                bbox_txt = tf.extractfile(m).read().decode()
            elif m.name == "CUB_200_2011/images.txt":
                images_txt = tf.extractfile(m).read().decode()
    assert all(b is not None for b in imgs), "missing test images in the archive"
    return imgs, bbox_txt, images_txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cub", required=True, help="CUB_200_2011.tgz or the extracted CUB_200_2011 folder")
    a = ap.parse_args()
    names = pickle.load(open(REPO / "data/birds/test/filenames.pickle", "rb"))
    imgs, bbox_txt, images_txt = read_cub(a.cub, names)
    id2file = dict(line.split() for line in images_txt.strip().splitlines())
    bbox = {}
    for line in bbox_txt.strip().splitlines():
        i, x, y, w, h = line.split()
        bbox[id2file[i][:-4]] = [int(float(v)) for v in (x, y, w, h)]
    arr = np.zeros((len(names), 256, 256, 3), np.uint8)
    full = np.zeros_like(arr)
    for i, (k, raw) in enumerate(zip(names, imgs)):
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        full[i] = np.asarray(img.resize((256, 256), Image.BICUBIC))
        W, H = img.size
        b = bbox[k]
        r = int(np.maximum(b[2], b[3]) * 0.75)          # same crop as datasets.get_imgs
        cx, cy = int((2 * b[0] + b[2]) / 2), int((2 * b[1] + b[3]) / 2)
        img = img.crop([max(0, cx - r), max(0, cy - r), min(W, cx + r), min(H, cy + r)])
        arr[i] = np.asarray(img.resize((256, 256), Image.BICUBIC))
    OUT.mkdir(exist_ok=True)
    np.save(OUT / "real_test_256.npy", arr)
    np.save(OUT / "real_test_256_uncropped.npy", full)
    print("saved", arr.shape, "(cropped and uncropped)")


if __name__ == "__main__":
    main()
