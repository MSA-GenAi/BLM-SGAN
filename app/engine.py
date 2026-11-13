"""
BLM-SGAN inference engine used by the web app.

It serves several generators side by side. Each model lives in checkpoints/birds/<id>/ with a netG_<epoch>.pth and
a model.json ({"name", "kind", "order", "description"}):

kind "blm"    - BLM-SGAN: the caption text goes to its own BERT-DAMSM text encoder (training/text_encoder.py),
                stored next to the generator as text_encoder.pth. Any English words work.
kind "ssagan" - SSA-GAN (Liao et al., CVPR 2022), conditioned on AttnGAN's DAMSM LSTM text encoder
                (DAMSMencoders/bird/text_encoder200.pth) over the CUB vocabulary (data/birds/captions.pickle).

Both use SSA-GAN's generator (model.NetG); every model keeps its checkpoint's own BatchNorm statistics.
"""
from __future__ import annotations

import base64
import io
import json
import os
import pickle
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import NetG  # noqa: E402  (project code)
from DAMSM import RNN_ENCODER  # noqa: E402  (project code)

NF = 64                   # generator width (SSA-GAN's NetG(64, 100))
Z_DIM = 100               # noise size
WORDS_NUM = 18            # SSA-GAN's DAMSM LSTM reads at most 18 words (miscc/config.py TEXT.WORDS_NUM)
MAX_IMAGES = 8
STAGES = [4, 8, 16, 32, 64, 128, 256]

MODELS_DIR = ROOT / "checkpoints" / "birds"
DEFAULT_VOCAB = ROOT / "data" / "birds" / "captions.pickle"
DAMSM_TEXT = ROOT / "DAMSMencoders" / "bird" / "text_encoder200.pth"
LOCAL_BERT = ROOT / "pretrained" / "bert-base-uncased"
OUTPUT_DIR = ROOT / "outputs"

_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE | re.MULTILINE | re.DOTALL)


def _inferno_lut() -> np.ndarray:
    try:
        import matplotlib
        matplotlib.use("Agg")
        cmap = matplotlib.colormaps["inferno"]
        return (cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    except Exception:  # fallback: plain grayscale
        g = np.arange(256, dtype=np.uint8)
        return np.stack([g, g, g], axis=1)


LUT = _inferno_lut()


def _png_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _to_uint8_image(t: torch.Tensor) -> np.ndarray:
    # [-1, 1] -> [0, 255], CHW -> HWC
    im = ((t.detach().float().cpu().clamp(-1, 1) + 1.0) * 127.5).round().byte().numpy()
    return np.transpose(im, (1, 2, 0))


def _mask_to_rgb(m: torch.Tensor) -> np.ndarray:
    idx = (m.detach().float().cpu().clamp(0, 1)[0] * 255).round().byte().numpy()
    return LUT[idx]


def tokenize(caption: str) -> list[str]:
    """Lower-case words, the form of the CUB captions (AttnGAN's preprocessing)."""
    cap = caption.replace("��", " ")
    out = []
    for t in _TOKEN_RE.findall(cap.lower()):
        t = t.encode("ascii", "ignore").decode("ascii")
        if t:
            out.append(t)
    return out


def _pick_device() -> torch.device:
    forced = os.environ.get("BLM_DEVICE")
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        try:  # make sure kernels exist for this GPU architecture
            x = torch.randn(1, 3, 8, 8, device="cuda")
            torch.nn.functional.conv2d(x, torch.randn(4, 3, 3, 3, device="cuda"))
            torch.cuda.synchronize()
            return torch.device("cuda")
        except Exception as exc:  # pragma: no cover
            print(f"[engine] CUDA present but unusable ({exc}); falling back to CPU")
    return torch.device("cpu")


def _epoch(p: Path) -> int:
    m = re.search(r"(\d+)", p.stem)
    return int(m.group(1)) if m else -1


def discover_models() -> list[dict]:
    """checkpoints/birds/<id>/{netG_*.pth, model.json}; BLM_NETG=<path> adds a single extra model."""
    specs = []
    if MODELS_DIR.is_dir() and not os.environ.get("BLM_ONLY_CUSTOM"):
        for d in sorted(p for p in MODELS_DIR.iterdir() if p.is_dir() and not p.name.startswith(".")):
            ckpts = sorted(d.glob("netG*.pth"), key=lambda p: (_epoch(p), p.stat().st_mtime))
            if not ckpts:
                continue
            meta = {}
            if (d / "model.json").is_file():
                try:
                    meta = json.loads((d / "model.json").read_text(encoding="utf-8"))
                except Exception as exc:
                    print(f"[engine] bad {d / 'model.json'}: {exc}")
            specs.append({"id": d.name, "name": meta.get("name", d.name), "kind": meta.get("kind", "blm"),
                          "order": meta.get("order", 100), "description": meta.get("description", ""),
                          "source": meta.get("source", ""), "ckpt": ckpts[-1],
                          "text_encoder": d / "text_encoder.pth"})
    env = os.environ.get("BLM_NETG")
    if env and Path(env).is_file():
        p = Path(env)
        specs.append({"id": "custom", "name": f"Custom ({p.name})", "kind": os.environ.get("BLM_KIND", "blm"),
                      "order": 0, "description": str(p), "source": str(p), "ckpt": p,
                      "text_encoder": Path(os.environ.get("BLM_TEXT_ENCODER", p.parent / "text_encoder.pth"))})
    return sorted(specs, key=lambda s: (s["order"], s["name"]))


def find_vocab() -> Path | None:
    env = os.environ.get("BLM_VOCAB")
    if env:
        p = Path(env)
        return p if p.is_file() else None
    for p in [DEFAULT_VOCAB, ROOT / "data" / "captions.pickle"]:
        if p.is_file():
            return p
    return None


def _load_netg(path: Path) -> NetG:
    try:
        sd = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        sd = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict):
        for key in ("netG", "state_dict", "model"):
            if key in sd and isinstance(sd[key], dict):
                sd = sd[key]
                break
    g = NetG(NF, Z_DIM)
    g.load_state_dict({(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}, strict=True)
    return g


class Engine:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.state = "idle"          # idle | loading | ready | error
        self.message = ""
        self.device = torch.device("cpu")
        self.lstm: RNN_ENCODER | None = None
        self.models: dict[str, dict] = {}      # id -> spec + "netG"
        self.default_model: str | None = None
        self.wordtoix: dict[str, int] = {}
        self.vocab_path: Path | None = None
        self.info: dict = {}

    # ------------------------------------------------------------------ load
    def load(self) -> None:
        with self.lock:
            self.state, self.message = "loading", "Preparing models"
            try:
                self._load()
                self.state, self.message = "ready", "Ready"
            except Exception as exc:
                self.state = "error"
                self.message = f"{type(exc).__name__}: {exc}"
                print("[engine] load failed:", self.message)

    def _load(self) -> None:
        t0 = time.time()
        self.device = _pick_device()
        bert_src = str(LOCAL_BERT) if (LOCAL_BERT / "config.json").is_file() else "bert-base-uncased"

        self.message = "Loading vocabulary"
        vocab = find_vocab()
        self.wordtoix = {}
        if vocab is not None:
            with open(vocab, "rb") as f:
                x = pickle.load(f)   # [train_captions, test_captions, ixtoword, wordtoix]
            self.wordtoix = dict(x[3])
        self.vocab_path = vocab

        specs = discover_models()
        if not specs:
            raise FileNotFoundError(f"No models found. Put netG_*.pth files in {MODELS_DIR}\\<name>\\ (see README.txt there).")

        self.lstm = None
        if any(s["kind"] == "ssagan" for s in specs):
            if not self.wordtoix or not DAMSM_TEXT.is_file():
                print("[engine] SSA-GAN models need captions.pickle and DAMSMencoders/bird/text_encoder200.pth; skipping them")
                specs = [s for s in specs if s["kind"] != "ssagan"]
            else:
                self.message = "Loading DAMSM LSTM text encoder"
                lstm = RNN_ENCODER(len(self.wordtoix), nhidden=256)
                lstm.load_state_dict(torch.load(DAMSM_TEXT, map_location="cpu", weights_only=True))
                self.lstm = lstm.to(self.device).eval()

        encoders: dict[Path, torch.nn.Module] = {}
        for s in [s for s in specs if s["kind"] == "blm"]:
            if not s["text_encoder"].is_file():
                print(f"[engine] {s['id']}: BLM-SGAN needs {s['text_encoder']}; skipping it")
                specs.remove(s)
                continue
            if s["text_encoder"] not in encoders:
                self.message = f"Loading text encoder of {s['name']}"
                from training.text_encoder import BertTextEncoder
                enc = BertTextEncoder.load(s["text_encoder"], bert=bert_src)
                encoders[s["text_encoder"]] = enc.to(self.device).eval()

        self.models = {}
        for s in specs:
            self.message = f"Loading {s['name']}"
            g = _load_netg(s["ckpt"]).to(self.device).eval()
            self.models[s["id"]] = {**s, "netG": g, "epoch": _epoch(s["ckpt"])}
            if s["kind"] == "blm":
                self.models[s["id"]]["text_enc"] = encoders[s["text_encoder"]]
        if not self.models:
            raise RuntimeError("No usable models (SSA-GAN needs the DAMSM files, BLM-SGAN a text_encoder.pth).")
        self.default_model = next(iter(self.models))
        self.info = {
            "device": self.device.type,
            "device_name": torch.cuda.get_device_name(0) if self.device.type == "cuda" else "CPU",
            "vocab_path": str(vocab) if vocab else None,
            "vocab_size": len(self.wordtoix),
            "bert": "bert-base-uncased (local copy)" if bert_src != "bert-base-uncased" else "bert-base-uncased",
            "params_text": sum(p.numel() for e in encoders.values() for p in e.parameters()),
            "params_generator": sum(p.numel() for p in next(iter(self.models.values()))["netG"].parameters()),
            "load_seconds": round(time.time() - t0, 1),
            "torch": torch.__version__,
        }
        print("[engine] ready:", json.dumps(self.info), "models:", list(self.models))

    # ------------------------------------------------------------- encoding
    def analyze(self, caption: str, kind: str | None = None) -> dict:
        tokens = tokenize(caption)
        if not self.wordtoix or kind == "blm":        # BLM-SGAN reads any words
            return {"tokens": [{"w": t, "known": True} for t in tokens], "used": tokens, "dropped": [], "truncated": False}
        known = [t for t in tokens if t in self.wordtoix]
        return {"tokens": [{"w": t, "known": t in self.wordtoix} for t in tokens], "used": known[:WORDS_NUM],
                "dropped": [t for t in tokens if t not in self.wordtoix], "truncated": len(known) > WORDS_NUM}

    def _ids(self, captions: list[str]) -> list[list[int]]:
        out = []
        for c in captions:
            ids = [self.wordtoix[t] for t in tokenize(c) if t in self.wordtoix][:WORDS_NUM]
            if not ids:
                raise ValueError(f'None of the words in "{c}" are in the bird vocabulary.')
            out.append(ids)
        return out

    def _encode_lstm(self, id_lists: list[list[int]]) -> torch.Tensor:
        assert self.lstm is not None
        lens = torch.tensor([len(c) for c in id_lists])
        order = torch.argsort(lens, descending=True)          # pack_padded_sequence needs sorted lengths
        rows = torch.zeros(len(id_lists), WORDS_NUM, dtype=torch.long)
        for k, c in enumerate(id_lists):
            rows[k, :len(c)] = torch.tensor(c, dtype=torch.long)
        with torch.no_grad():
            _, sent = self.lstm(rows[order].to(self.device), lens[order].to(self.device),
                                self.lstm.init_hidden(len(id_lists)))
        inv = torch.empty_like(order)
        inv[order] = torch.arange(len(order))
        return sent[inv.to(self.device)].detach()

    def _encode(self, model: dict, captions: list[str], mode: str) -> torch.Tensor:
        if model["kind"] == "ssagan":
            return self._encode_lstm(self._ids(captions))
        texts = [" ".join(tokenize(c)) for c in captions]   # BLM-SGAN: the same text form as in training
        if not all(texts):
            raise ValueError("Caption is empty.")
        with torch.no_grad():
            return model["text_enc"](texts)[1].detach()

    # ----------------------------------------------------------- generation
    def _noise(self, seeds: list[int], scale: float) -> torch.Tensor:
        zs = [torch.randn(1, Z_DIM, generator=torch.Generator().manual_seed(int(s))) for s in seeds]
        return (torch.cat(zs, 0) * float(scale)).to(self.device)

    def _run(self, model: dict, sent: torch.Tensor, seeds: list[int], scale: float):
        with torch.no_grad():
            return model["netG"](self._noise(seeds, scale), sent)

    def _pack(self, imgs, masks, i: int, seed: int) -> tuple[dict, np.ndarray]:
        arr = _to_uint8_image(imgs[i])
        return {"seed": seed, "image": _png_b64(arr),
                "masks": [{"size": s, "png": _png_b64(_mask_to_rgb(masks[k][i]))} for k, s in enumerate(STAGES)]}, arr

    def _model(self, model_id: str | None) -> dict:
        self._require_ready()
        mid = model_id or self.default_model
        if mid not in self.models:
            raise ValueError(f"Unknown model '{mid}'.")
        return self.models[mid]

    @staticmethod
    def _resolve_mode(model: dict, mode: str | None = None) -> str:
        """The text encoder a model reads captions with (the mode argument is kept for API compatibility)."""
        return "lstm" if model["kind"] == "ssagan" else "bert"

    def generate(self, caption: str, n: int, seed: int, scale: float = 1.0, mode: str | None = None,
                 save: bool = True, model_id: str | None = None) -> dict:
        model = self._model(model_id)
        mode = self._resolve_mode(model, mode)
        n = max(1, min(int(n), MAX_IMAGES))
        seeds = [int(seed) + i for i in range(n)]
        t0 = time.time()
        with self.lock:
            sent = self._encode(model, [caption], mode).repeat(n, 1)
            imgs, masks = self._run(model, sent, seeds, scale)
            out, arrays = [], []
            for i, s in enumerate(seeds):
                item, arr = self._pack(imgs, masks, i, s)
                out.append(item)
                arrays.append(arr)
        ms = int((time.time() - t0) * 1000)
        saved = self._save(caption, seeds, arrays, mode, scale, model) if save else None
        return {"caption": caption, "mode": mode, "scale": scale, "model": self._public(model), "images": out,
                "analysis": self.analyze(caption, model["kind"]), "ms": ms, "saved_to": saved}

    def compare(self, captions: list[str], n_noise: int, seed: int, scale: float = 1.0,
                mode: str | None = None, model_id: str | None = None) -> dict:
        model = self._model(model_id)
        mode = self._resolve_mode(model, mode)
        captions = [c.strip() for c in captions if c and c.strip()][:8]
        if not captions:
            raise ValueError("Add at least one caption.")
        n_noise = max(1, min(int(n_noise), 4))
        seeds = [int(seed) + i for i in range(n_noise)]
        t0 = time.time()
        rows = []
        with self.lock:
            sents = self._encode(model, captions, mode)
            for r, cap in enumerate(captions):
                imgs, masks = self._run(model, sents[r:r + 1].repeat(n_noise, 1), seeds, scale)
                rows.append({"label": cap, "caption": cap, "analysis": self.analyze(cap, model["kind"]),
                             "images": [self._pack(imgs, masks, i, s)[0] for i, s in enumerate(seeds)]})
        return {"by": "captions", "rows": rows, "seeds": seeds, "mode": mode, "scale": scale,
                "model": self._public(model), "ms": int((time.time() - t0) * 1000)}

    def compare_models(self, caption: str, model_ids: list[str] | None, n_noise: int, seed: int,
                       scale: float = 1.0) -> dict:
        self._require_ready()
        ids = [m for m in (model_ids or list(self.models)) if m in self.models][:6]
        if not ids:
            raise ValueError("Choose at least one model.")
        n_noise = max(1, min(int(n_noise), 4))
        seeds = [int(seed) + i for i in range(n_noise)]
        t0 = time.time()
        rows = []
        with self.lock:
            for mid in ids:
                model = self.models[mid]
                mode = self._resolve_mode(model, None)
                imgs, masks = self._run(model, self._encode(model, [caption], mode).repeat(n_noise, 1), seeds, scale)
                rows.append({"label": model["name"], "caption": caption, "model": self._public(model), "mode": mode,
                             "images": [self._pack(imgs, masks, i, s)[0] for i, s in enumerate(seeds)]})
        return {"by": "models", "caption": caption, "analysis": self.analyze(caption), "rows": rows,
                "seeds": seeds, "scale": scale, "ms": int((time.time() - t0) * 1000)}

    # --------------------------------------------------------------- helpers
    def _require_ready(self) -> None:
        if self.state != "ready":
            raise RuntimeError(f"Model is not ready ({self.state}): {self.message}")

    @staticmethod
    def _public(model: dict) -> dict:
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in model.items()
                if k not in ("netG", "ckpt", "text_enc", "text_encoder")} | {"checkpoint": model["ckpt"].name}

    def _save(self, caption, seeds, arrays, mode, scale, model) -> str | None:
        try:
            slug = re.sub(r"[^a-z0-9]+", "-", caption.lower()).strip("-")[:48] or "caption"
            now = datetime.now()
            d = OUTPUT_DIR / now.strftime("%Y-%m-%d") / f"{now.strftime('%H%M%S')}_{model['id']}_{slug}"
            d.mkdir(parents=True, exist_ok=True)
            for s, arr in zip(seeds, arrays):
                Image.fromarray(arr).save(d / f"seed_{s}.png")
            meta = {"caption": caption, "seeds": seeds, "encoding": mode, "noise_scale": scale,
                    "model": model["name"], "checkpoint": str(model["ckpt"].relative_to(ROOT)) if model["ckpt"].is_relative_to(ROOT) else str(model["ckpt"]),
                    "time": now.isoformat()}
            (d / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            return str(d.relative_to(ROOT))
        except Exception as exc:  # saving is best-effort
            print("[engine] could not save outputs:", exc)
            return None

    def status(self) -> dict:
        return {"state": self.state, "message": self.message, "max_images": MAX_IMAGES, "words_num": WORDS_NUM,
                "vocab": bool(self.wordtoix), "default_model": self.default_model,
                "models": [self._public(m) for m in self.models.values()], "info": self.info}
