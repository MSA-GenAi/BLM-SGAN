@echo off
setlocal
cd /d "%~dp0"

set "PY="
if exist "%~dp0.venv\Scripts\python.exe" set "PY=%~dp0.venv\Scripts\python.exe"
if not defined PY if exist "%~dp0..\.venv\Scripts\python.exe" set "PY=%~dp0..\.venv\Scripts\python.exe"

if not defined PY (
  echo [setup] Creating virtual environment in .venv ...
  python -m venv "%~dp0.venv" || goto :fail
  set "PY=%~dp0.venv\Scripts\python.exe"
  "%~dp0.venv\Scripts\python.exe" -m pip install --upgrade pip
  "%~dp0.venv\Scripts\python.exe" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 || goto :fail
  "%~dp0.venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :fail
)

if not exist "%~dp0pretrained\bert-base-uncased\model.safetensors" (
  echo [setup] Downloading BERT base uncased for offline use ...
  "%PY%" -c "from huggingface_hub import snapshot_download as d; d('google-bert/bert-base-uncased', local_dir='pretrained/bert-base-uncased', allow_patterns=['config.json','vocab.txt','tokenizer_config.json','tokenizer.json','model.safetensors'])"
)

set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
if not exist "%~dp0pretrained\bert-base-uncased\model.safetensors" (
  set HF_HUB_OFFLINE=
  set TRANSFORMERS_OFFLINE=
)

if not defined BLM_NO_BROWSER start "" /b cmd /c "timeout /t 4 /nobreak >nul & start http://127.0.0.1:8000"
"%PY%" -m app.server
goto :eof

:fail
echo.
echo Setup failed. See the messages above.
pause
