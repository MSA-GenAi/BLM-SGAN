# BLM-SGAN Studio, public mode (app/server.py, BLM_PUBLIC=1), on CPU, port 7860.
# Build from the folder that deploy/build_bundle.py assembles:  docker build -t blm-sgan deploy/_bundle
FROM python:3.12-slim

RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    BLM_PUBLIC=1 \
    BLM_DEVICE=cpu \
    BLM_HOST=0.0.0.0 \
    BLM_PORT=7860
WORKDIR /home/user/app

RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

USER user
# BERT is baked into the image, so a Space waking from sleep starts without downloading it
RUN python -c "from transformers import BertModel, BertTokenizer, BertTokenizerFast; \
BertModel.from_pretrained('bert-base-uncased'); BertTokenizer.from_pretrained('bert-base-uncased'); \
BertTokenizerFast.from_pretrained('bert-base-uncased')"
ENV HF_HUB_OFFLINE=1

COPY --chown=user . .
EXPOSE 7860
CMD ["python", "-m", "app.server"]
