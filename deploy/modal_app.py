"""BLM-SGAN Studio, public, on Modal (free monthly credits, no card): https://<workspace>--blm-sgan.modal.run

    python deploy/build_bundle.py --out deploy/_bundle     # the app files (public mode)
    modal deploy deploy/modal_app.py

The app runs in public mode (no How-it-works page). It sleeps after SLEEP_AFTER seconds with no requests and wakes
on the next visit, which then waits ~30-60 s while the models load. max_containers=1 caps the cost: at most one
2-CPU container ever runs, so the free monthly credits last far beyond a demo's use.
"""
from pathlib import Path

import modal

BUNDLE = Path(__file__).resolve().parent / "_bundle"
SLEEP_AFTER = 15 * 60
BERT = ("from transformers import BertModel, BertTokenizer, BertTokenizerFast; "
        "BertModel.from_pretrained('bert-base-uncased'); BertTokenizer.from_pretrained('bert-base-uncased'); "
        "BertTokenizerFast.from_pretrained('bert-base-uncased')")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "torchvision", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install_from_requirements(str(BUNDLE / "requirements.txt"))
    .run_commands(f'python -c "{BERT}"')          # BERT is part of the image: no download when it wakes up
    .env({"BLM_PUBLIC": "1", "BLM_DEVICE": "cpu", "HF_HUB_OFFLINE": "1"})
    .add_local_dir(BUNDLE, remote_path="/app", ignore=["Dockerfile", "**/__pycache__"])
)
app = modal.App("blm-sgan", image=image)


@app.function(cpu=2.0, memory=4096, min_containers=0, max_containers=1, scaledown_window=SLEEP_AFTER, timeout=300)
@modal.concurrent(max_inputs=16)
@modal.asgi_app(label="blm-sgan")
def web():
    import os
    import sys
    os.chdir("/app")
    sys.path.insert(0, "/app")
    from app import server
    server._load_async()      # start loading the models now; the page shows progress until they are ready
    return server.app
