#!/usr/bin/env bash
# Evaluate every model in checkpoints/birds/ with the same protocol.
# Prerequisite (once): python evaluation/prepare_real.py --cub <CUB_200_2011.tgz | CUB_200_2011 folder>
# Each model: generate images (only the generator on the GPU), then score them (only the judges on the GPU).
# Usage: PY=python bash evaluation/run_all.sh
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}
export PYTHONIOENCODING=utf-8 PYTHONWARNINGS=ignore

run() {  # tag model ckpt
  echo "===== $1  ($3)  $(date +%T)"
  [ -f "evaluation/cache/gen_$1.json" ] || $PY -u evaluation/evaluate.py --model "$2" --ckpt "$3" --tag "$1" --stage gen
  $PY -u evaluation/evaluate.py --model "$2" --ckpt "$3" --tag "$1" --stage judge
}

[ -f evaluation/results/real_images.json ] || $PY -u evaluation/evaluate.py --real
for d in checkpoints/birds/*/; do
  id=$(basename "$d")
  ck=$(ls "$d"netG_*.pth 2>/dev/null | sort -V | tail -1)
  [ -n "$ck" ] || continue
  kind=$($PY -c "import json,sys; print(json.load(open(sys.argv[1])).get('kind','blm'))" "${d}model.json" 2>/dev/null || echo blm)
  run "${id}_$(basename "$ck" .pth | sed 's/netG_//')" "$kind" "$ck"
done
$PY -u evaluation/inception_score.py   # adds IS to every results/*.json from the cached Inception features
echo "ALL DONE $(date +%T)"
