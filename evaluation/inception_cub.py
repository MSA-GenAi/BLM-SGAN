"""StackGAN's CUB-fine-tuned Inception-v3, ported to PyTorch: the classifier CUB papers use for IS.

StackGAN, AttnGAN, DM-GAN, DF-GAN, SSA-GAN and RAT-GAN report Inception Score with this network: Inception-v3
fine-tuned on CUB (50 classes + an unused background class 0), TF-slim checkpoint
`inception_finetuned_models/birds_valid299/model.ckpt` from StackGAN's Google Drive. This file rebuilds the
same graph in PyTorch (TF-slim `inception_v3`, eval mode: BatchNorm with beta only, eps 1e-3, moving averages;
'SAME' average pooling that ignores the padding) so the score runs on the GPU next to FID.

Convert the checkpoint once (needs TensorFlow, any 2.x):
    python evaluation/inception_cub.py convert <.../birds_valid299/model.ckpt> pretrained/inception_cub.pth
Score images (N x H x W x 3 uint8, any size: resized to 299 like the papers):
    from evaluation.inception_cub import load, probabilities, inception_score
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent / "pretrained" / "inception_cub.pth"


class Conv(nn.Module):
    """TF-slim conv2d + batch_norm(scale=False) + ReLU."""

    def __init__(self, cin, cout, k, stride=1, pad=0):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride, pad, bias=False)
        self.beta = nn.Parameter(torch.zeros(cout))
        self.register_buffer("mean", torch.zeros(cout))
        self.register_buffer("var", torch.ones(cout))

    def forward(self, x):
        x = self.conv(x)
        x = (x - self.mean[:, None, None]) * torch.rsqrt(self.var[:, None, None] + 1e-3) + self.beta[:, None, None]
        return F.relu(x)


def avg3(x):  # TF avg_pool 3x3 stride 1 'SAME': padded cells are not counted
    return F.avg_pool2d(x, 3, 1, 1, count_include_pad=False)


class A(nn.Module):  # mixed_35x35x*
    def __init__(self, cin, pool):
        super().__init__()
        self.b1 = Conv(cin, 64, 1)
        self.b5 = nn.Sequential(Conv(cin, 48, 1), Conv(48, 64, 5, pad=2))
        self.b3 = nn.Sequential(Conv(cin, 64, 1), Conv(64, 96, 3, pad=1), Conv(96, 96, 3, pad=1))
        self.bp = Conv(cin, pool, 1)

    def forward(self, x):
        return torch.cat([self.b1(x), self.b5(x), self.b3(x), self.bp(avg3(x))], 1)


class B(nn.Module):  # mixed_17x17x768a
    def __init__(self):
        super().__init__()
        self.b3 = Conv(288, 384, 3, stride=2)
        self.b3d = nn.Sequential(Conv(288, 64, 1), Conv(64, 96, 3, pad=1), Conv(96, 96, 3, stride=2))

    def forward(self, x):
        return torch.cat([self.b3(x), self.b3d(x), F.max_pool2d(x, 3, 2)], 1)


class C(nn.Module):  # mixed_17x17x768b-e
    def __init__(self, c7):
        super().__init__()
        self.b1 = Conv(768, 192, 1)
        self.b7 = nn.Sequential(Conv(768, c7, 1), Conv(c7, c7, (1, 7), pad=(0, 3)), Conv(c7, 192, (7, 1), pad=(3, 0)))
        self.b7d = nn.Sequential(Conv(768, c7, 1), Conv(c7, c7, (7, 1), pad=(3, 0)), Conv(c7, c7, (1, 7), pad=(0, 3)),
                                 Conv(c7, c7, (7, 1), pad=(3, 0)), Conv(c7, 192, (1, 7), pad=(0, 3)))
        self.bp = Conv(768, 192, 1)

    def forward(self, x):
        return torch.cat([self.b1(x), self.b7(x), self.b7d(x), self.bp(avg3(x))], 1)


class D(nn.Module):  # mixed_17x17x1280a
    def __init__(self):
        super().__init__()
        self.b3 = nn.Sequential(Conv(768, 192, 1), Conv(192, 320, 3, stride=2))
        self.b7 = nn.Sequential(Conv(768, 192, 1), Conv(192, 192, (1, 7), pad=(0, 3)),
                                Conv(192, 192, (7, 1), pad=(3, 0)), Conv(192, 192, 3, stride=2))

    def forward(self, x):
        return torch.cat([self.b3(x), self.b7(x), F.max_pool2d(x, 3, 2)], 1)


class E(nn.Module):  # mixed_8x8x2048a/b (average pooling in both, unlike the 2015 ImageNet graph)
    def __init__(self, cin):
        super().__init__()
        self.b1 = Conv(cin, 320, 1)
        self.b3 = Conv(cin, 384, 1)
        self.b3a, self.b3b = Conv(384, 384, (1, 3), pad=(0, 1)), Conv(384, 384, (3, 1), pad=(1, 0))
        self.b3d = nn.Sequential(Conv(cin, 448, 1), Conv(448, 384, 3, pad=1))
        self.b3da, self.b3db = Conv(384, 384, (1, 3), pad=(0, 1)), Conv(384, 384, (3, 1), pad=(1, 0))
        self.bp = Conv(cin, 192, 1)

    def forward(self, x):
        b3, b3d = self.b3(x), self.b3d(x)
        return torch.cat([self.b1(x), self.b3a(b3), self.b3b(b3), self.b3da(b3d), self.b3db(b3d), self.bp(avg3(x))], 1)


class InceptionCUB(nn.Module):
    def __init__(self, num_classes=51):
        super().__init__()
        self.stem = nn.ModuleList([Conv(3, 32, 3, stride=2), Conv(32, 32, 3), Conv(32, 64, 3, pad=1),
                                   Conv(64, 80, 1), Conv(80, 192, 3)])
        self.blocks = nn.Sequential(A(192, 32), A(256, 64), A(288, 64), B(), C(128), C(160), C(160), C(192), D(),
                                    E(1280), E(2048))
        self.fc = nn.Linear(2048, num_classes)

    def features(self, x):
        """x: N x 3 x 299 x 299 in [-1, 1] -> N x 2048 (pool of mixed_8x8x2048b)."""
        s = self.stem
        x = s[2](s[1](s[0](x)))
        x = F.max_pool2d(x, 3, 2)
        x = s[4](s[3](x))
        x = F.max_pool2d(x, 3, 2)
        return self.blocks(x).mean((2, 3))

    def forward(self, x):
        return self.fc(self.features(x))


# ------------------------------------------------------------------ TF checkpoint -> PyTorch
def _tf_conv_names():
    """TF-slim variable scopes of every conv, in the order the PyTorch modules are created."""
    names = [f"conv{i}" for i in range(5)]
    def a(s): return [f"{s}/branch1x1/Conv", f"{s}/branch5x5/Conv", f"{s}/branch5x5/Conv_1",
                      f"{s}/branch3x3dbl/Conv", f"{s}/branch3x3dbl/Conv_1", f"{s}/branch3x3dbl/Conv_2",
                      f"{s}/branch_pool/Conv"]
    def c(s): return ([f"{s}/branch1x1/Conv"] + [f"{s}/branch7x7/Conv" + t for t in ["", "_1", "_2"]] +
                      [f"{s}/branch7x7dbl/Conv" + t for t in ["", "_1", "_2", "_3", "_4"]] + [f"{s}/branch_pool/Conv"])
    def e(s): return [f"{s}/branch1x1/Conv", f"{s}/branch3x3/Conv", f"{s}/branch3x3/Conv_1", f"{s}/branch3x3/Conv_2",
                      f"{s}/branch3x3dbl/Conv", f"{s}/branch3x3dbl/Conv_1", f"{s}/branch3x3dbl/Conv_2",
                      f"{s}/branch3x3dbl/Conv_3", f"{s}/branch_pool/Conv"]
    names += a("mixed_35x35x256a") + a("mixed_35x35x288a") + a("mixed_35x35x288b")
    names += ["mixed_17x17x768a/branch3x3/Conv", "mixed_17x17x768a/branch3x3dbl/Conv",
              "mixed_17x17x768a/branch3x3dbl/Conv_1", "mixed_17x17x768a/branch3x3dbl/Conv_2"]
    for s in "bcde":
        names += c(f"mixed_17x17x768{s}")
    names += ["mixed_17x17x1280a/branch3x3/Conv", "mixed_17x17x1280a/branch3x3/Conv_1",
              "mixed_17x17x1280a/branch7x7x3/Conv", "mixed_17x17x1280a/branch7x7x3/Conv_1",
              "mixed_17x17x1280a/branch7x7x3/Conv_2", "mixed_17x17x1280a/branch7x7x3/Conv_3"]
    names += e("mixed_8x8x2048a") + e("mixed_8x8x2048b")
    return names


def _conv_modules(m):
    """Conv modules in the order of _tf_conv_names (E's forward order differs from creation order)."""
    out = list(m.stem)
    for blk in m.blocks:
        if isinstance(blk, E):
            out += [blk.b1, blk.b3, blk.b3a, blk.b3b, blk.b3d[0], blk.b3d[1], blk.b3da, blk.b3db, blk.bp]
        else:
            out += [mod for mod in blk.modules() if isinstance(mod, Conv)]
    return out


def convert(ckpt, out):
    import tensorflow as tf
    r = tf.train.load_checkpoint(ckpt)
    keys = r.get_variable_to_shape_map()

    def get(name):  # the moving-average shadow when there is one, as the papers' eval code restores it
        ema = name + "/ExponentialMovingAverage"
        return torch.from_numpy(r.get_tensor(ema if ema in keys else name).copy())

    m = InceptionCUB()
    names, mods = _tf_conv_names(), _conv_modules(m)
    assert len(names) == len(mods) == 94, (len(names), len(mods))
    used = set()
    for n, mod in zip(names, mods):
        w = get(f"{n}/weights")                       # kh, kw, cin, cout
        assert tuple(w.shape[:2]) == mod.conv.kernel_size and w.shape[3] == mod.conv.out_channels, n
        mod.conv.weight.data = w.permute(3, 2, 0, 1).contiguous()
        mod.beta.data = get(f"{n}/BatchNorm/beta")
        mod.mean.data = get(f"{n}/BatchNorm/moving_mean")
        mod.var.data = get(f"{n}/BatchNorm/moving_variance")
        used |= {f"{n}/weights", f"{n}/BatchNorm/beta", f"{n}/BatchNorm/moving_mean", f"{n}/BatchNorm/moving_variance"}
    m.fc.weight.data = get("logits/logits/weights").T.contiguous()
    m.fc.bias.data = get("logits/logits/biases")
    used |= {"logits/logits/weights", "logits/logits/biases"}
    left = sorted(k for k in keys if k.split("/ExponentialMovingAverage")[0] not in used
                  and not k.startswith(("aux_logits", "tower_", "beta1_power", "beta2_power", "global_step"))
                  and not k.endswith(("/Adam", "/Adam_1")))
    print("checkpoint variables not used (expected: none):", left)
    torch.save(m.state_dict(), out)
    print("saved", out)


# ------------------------------------------------------------------ scoring
def load(path=DEFAULT_WEIGHTS, device="cpu"):
    m = InceptionCUB()
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return m.to(device).eval()


@torch.no_grad()
def probabilities(model, images, device, bs=50):
    """images: N x H x W x 3 uint8 (numpy, may be a memmap) -> N x 50 class probabilities (background dropped)."""
    out = []
    for i in range(0, len(images), bs):
        x = torch.from_numpy(np.array(images[i:i + bs])).to(device).permute(0, 3, 1, 2).float()
        if x.shape[-1] != 299:
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        x = x / 127.5 - 1.0
        out.append(torch.softmax(model(x)[:, 1:].double(), 1).cpu())
    return torch.cat(out)


def inception_score(p, splits=10, seed=0):
    """StackGAN's formula: exp(E KL(p(y|x) || p(y))) per split, mean and std over splits (shuffled order)."""
    p = p[torch.randperm(len(p), generator=torch.Generator().manual_seed(seed))]
    scores, n = [], len(p)
    for k in range(splits):
        part = p[k * n // splits:(k + 1) * n // splits]
        kl = (part * (torch.log(part) - torch.log(part.mean(0, keepdim=True)))).sum(1).mean()
        scores.append(float(kl.exp()))
    return float(np.mean(scores)), float(np.std(scores))


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "convert":
        convert(sys.argv[2], sys.argv[3])
    else:
        print(__doc__)
