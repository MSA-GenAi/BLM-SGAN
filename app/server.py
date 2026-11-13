"""
BLM-SGAN Studio - web server.

Run from the repository root:
    python -m app.server            (or double-click start_app.bat)
then open http://127.0.0.1:8000

BLM_PUBLIC=1 runs the public version (the Hugging Face Space): no How-it-works page, no evaluation results,
no reload, nothing saved to disk, and only names and descriptions of the models in /api/status.
"""
from __future__ import annotations

import json
import os
import re
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .engine import Engine, MAX_IMAGES

STATIC = Path(__file__).resolve().parent / "static"
METRICS_DIR = Path(__file__).resolve().parent.parent / "evaluation" / "results"
PUBLIC = os.environ.get("BLM_PUBLIC", "").lower() in ("1", "true", "yes")

engine = Engine()


def _public_index() -> str:
    """index.html without the How-it-works tab and page."""
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    html = re.sub(r'\n[ \t]*<button role="tab"[^>]*id="tab-about"[^>]*>.*?</button>', "", html, flags=re.S)
    html, n = re.subn(r'\n[ \t]*<section class="view about".*?</section>(?=\s*</main>)', "", html, flags=re.S)
    assert n == 1, "index.html changed: the How-it-works section was not found"
    return html


_load_started = False


def _load_async(force: bool = False) -> None:
    """Load the models in the background, once (a host may start it before the ASGI lifespan does);
    force=True reloads (the Reload button)."""
    global _load_started
    if _load_started and not force:
        return
    _load_started = True
    threading.Thread(target=engine.load, daemon=True).start()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _load_async()          # load models in the background; the page shows progress
    yield


app = FastAPI(title="BLM-SGAN Studio", docs_url=None if PUBLIC else "/api/docs", redoc_url=None,
              openapi_url=None if PUBLIC else "/openapi.json", lifespan=lifespan)
INDEX_PUBLIC = _public_index() if PUBLIC else None


class GenerateIn(BaseModel):
    caption: str = Field(..., max_length=500)
    n: int = Field(4, ge=1, le=MAX_IMAGES)
    seed: int = Field(0, ge=0, le=2**31 - 1)
    scale: float = Field(1.0, ge=0.0, le=2.0)
    mode: str | None = None
    save: bool = True
    model: str | None = None


class CompareIn(BaseModel):
    captions: list[str] = Field(..., min_length=1, max_length=8)
    n_noise: int = Field(3, ge=1, le=4)
    seed: int = Field(0, ge=0, le=2**31 - 1)
    scale: float = Field(1.0, ge=0.0, le=2.0)
    mode: str | None = None
    model: str | None = None


class CompareModelsIn(BaseModel):
    caption: str = Field(..., max_length=500)
    models: list[str] | None = Field(None, max_length=6)
    n_noise: int = Field(3, ge=1, le=4)
    seed: int = Field(0, ge=0, le=2**31 - 1)
    scale: float = Field(1.0, ge=0.0, le=2.0)


class AnalyzeIn(BaseModel):
    caption: str = Field("", max_length=500)


PUBLIC_MODEL_KEYS = ("id", "name", "kind", "order", "description")


def _trim(res):
    """Public site: every model dict in a response keeps only its name and description fields."""
    if not PUBLIC:
        return res
    if isinstance(res, dict):
        return {k: ({f: v.get(f) for f in PUBLIC_MODEL_KEYS} if k == "model" and isinstance(v, dict) else _trim(v))
                for k, v in res.items()}
    if isinstance(res, list):
        return [_trim(v) for v in res]
    return res


def _fail(exc: Exception) -> HTTPException:
    code = 400 if isinstance(exc, ValueError) else 503
    return HTTPException(status_code=code, detail=str(exc))


@app.get("/api/status")
def status():
    st = engine.status()
    if PUBLIC:   # names and descriptions only: no checkpoint files, epochs, paths or machine details
        st = {k: v for k, v in st.items() if k != "info"} | {"public": True, "models": [
            {k: m.get(k) for k in PUBLIC_MODEL_KEYS} for m in st["models"]]}
    return st


@app.get("/api/vocab")
def vocab():
    return {"words": sorted(engine.wordtoix.keys())}


@app.post("/api/analyze")
def analyze(body: AnalyzeIn):
    return engine.analyze(body.caption)


@app.post("/api/generate")
def generate(body: GenerateIn):
    try:
        return _trim(engine.generate(body.caption, body.n, body.seed, body.scale, body.mode,
                                     body.save and not PUBLIC, body.model))
    except (ValueError, RuntimeError) as exc:
        raise _fail(exc)


@app.post("/api/compare")
def compare(body: CompareIn):
    try:
        return _trim(engine.compare(body.captions, body.n_noise, body.seed, body.scale, body.mode, body.model))
    except (ValueError, RuntimeError) as exc:
        raise _fail(exc)


@app.post("/api/compare_models")
def compare_models(body: CompareModelsIn):
    try:
        return _trim(engine.compare_models(body.caption, body.models, body.n_noise, body.seed, body.scale))
    except (ValueError, RuntimeError) as exc:
        raise _fail(exc)


@app.get("/api/metrics")
def metrics():
    """Evaluation results for the How-it-works page, if present: evaluation/results/*.json (main table),
    language_tests/*.json (rare words, long captions) and dfgan_protocol/*.json (FID on the DF-GAN / GALIP scale)."""
    if PUBLIC:
        raise HTTPException(status_code=404, detail="Not found")

    def read(folder: Path) -> list:
        out = []
        for p in sorted(folder.glob("*.json")) if folder.is_dir() else []:
            try:
                out.append(json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                pass
        return out
    return {"results": read(METRICS_DIR), "language_tests": read(METRICS_DIR / "language_tests"),
            "paper_scale": read(METRICS_DIR / "dfgan_protocol")}


@app.post("/api/reload")
def reload_model():
    if PUBLIC:
        raise HTTPException(status_code=403, detail="Not available on the public site")
    if engine.state == "loading":
        return JSONResponse({"ok": False, "detail": "Already loading"}, status_code=409)
    engine.state, engine.message = "loading", "Reloading"
    _load_async(force=True)
    return {"ok": True}


@app.get("/")
def index():
    return HTMLResponse(INDEX_PUBLIC) if PUBLIC else FileResponse(STATIC / "index.html")


@app.get("/static/index.html")   # before the static mount: the public site never serves the full page
def static_index():
    return index()


app.mount("/static", StaticFiles(directory=STATIC), name="static")


def main() -> None:
    import uvicorn
    host = os.environ.get("BLM_HOST", "127.0.0.1")
    port = int(os.environ.get("BLM_PORT", "8000"))
    print(f"\n  BLM-SGAN Studio  ->  http://{host}:{port}\n")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
