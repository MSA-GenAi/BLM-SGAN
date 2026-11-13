"""Assemble the public web app (public mode) into one folder, ready for deploy/modal_app.py or a Docker host.

    python deploy/build_bundle.py --out deploy/_bundle

The bundle gets only what the app needs: app/, the model code it imports, the two models in checkpoints/birds/
(BLM-SGAN and SSA-GAN), AttnGAN's DAMSM text encoder for SSA-GAN, the CUB vocabulary, and deploy/docker/
(Dockerfile, requirements.txt). BERT is downloaded when the image is built. Every copied file is compared with its
source, and the model files with their known MD5, so a damaged copy is never published.
"""
import argparse
import hashlib
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DOCKER_FILES = REPO / "deploy" / "docker"
CODE = ["app/__init__.py", "app/engine.py", "app/server.py", "app/static", "model.py", "DAMSM.py",
        "miscc/__init__.py", "miscc/config.py", "sync_batchnorm", "training/text_encoder.py"]
DATA = ["checkpoints/birds/blm_sgan/model.json", "checkpoints/birds/blm_sgan/netG_555.pth",
        "checkpoints/birds/blm_sgan/text_encoder.pth",
        "checkpoints/birds/ssagan/model.json", "checkpoints/birds/ssagan/netG_590.pth",
        "DAMSMencoders/bird/text_encoder200.pth", "data/birds/captions.pickle"]
# known-good checksums (this PC has corrupted large files before): never publish a damaged generator
MD5 = {"checkpoints/birds/blm_sgan/netG_555.pth": "273af05c6d8ec294ea0757089b0e966d",
       "checkpoints/birds/blm_sgan/text_encoder.pth": "bca18cfe95c4d137755ba5eec1e37a23",
       "checkpoints/birds/ssagan/netG_590.pth": "527c795701d70f1f652c23e885c7d23c",
       "DAMSMencoders/bird/text_encoder200.pth": "11fe9392efb4f231fd29f2cf77ff357e",
       "data/birds/captions.pickle": "c1cad6c0b32e6c3ce698cf01eaab2527"}


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def copy_verified(src: Path, dst: Path, tries: int = 3) -> None:
    """Copy and compare MD5s: this PC has corrupted large copies (same size, different bytes), so retry."""
    want = md5(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, tries + 1):
        shutil.copy2(src, dst)
        if md5(dst) == want:
            return
        print(f"  damaged copy of {src.name} (attempt {attempt}), copying again", flush=True)
    sys.exit(f"could not copy {src} intact in {tries} attempts")


def assemble(out: Path) -> None:
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} is not empty")
    out.mkdir(parents=True, exist_ok=True)
    skip = {"__pycache__", "unittest.py"}
    for rel in CODE + DATA:
        src = REPO / rel
        files = [p for p in src.rglob("*") if p.is_file()] if src.is_dir() else [src]
        for f in files:
            if skip & set(f.relative_to(REPO).parts) or f.suffix == ".pyc":
                continue
            copy_verified(f, out / f.relative_to(REPO))
    for f in DOCKER_FILES.iterdir():  # LF endings: a CRLF after a Dockerfile line continuation breaks the build
        (out / f.name).write_bytes(f.read_bytes().replace(b"\r\n", b"\n"))
    for rel, want in MD5.items():
        got = md5(out / rel)
        if got != want:
            sys.exit(f"checksum mismatch for {rel}: {got} (expected {want}) - not publishing a damaged file")
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"assembled {sum(1 for p in out.rglob('*') if p.is_file())} files, {size / 2**20:.0f} MB, in {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="empty folder to assemble the app in")
    assemble(Path(ap.parse_args().out))


if __name__ == "__main__":
    main()
