#!/bin/bash
set -Eeuo pipefail

echo "===> Versions"
python --version || true
python -m pip --version || true
node -v || true
npm -v || true

echo "===> Backend deps"
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt

# Si en algún momento vuelves a usar transformers, reactiva este bloque.
# echo "===> Pre-caching FinBERT (optional)"
# export HF_HOME="${HOME}/.cache/huggingface"
# python - <<'PY'
# try:
#     from transformers import AutoTokenizer, AutoModel
#     AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
#     AutoModel.from_pretrained("yiyanghkust/finbert-tone")
#     print("[OK] FinBERT precargado")
# except Exception as e:
#     print("[WARN] No se pudo precargar FinBERT:", e)
# PY

echo "===> Frontend install"
if [ -f frontend/package-lock.json ]; then
  npm ci --prefix frontend
else
  npm install --prefix frontend
fi

echo "===> Frontend build"
export NODE_OPTIONS="--max-old-space-size=2048"
npm run build --prefix frontend

echo "===> Copy dist -> backend/static"
rm -rf backend/static
mkdir -p backend/static
cp -r frontend/dist/* backend/static/

# sanity checks
if [ ! -f backend/static/index.html ]; then
  echo "[ERROR] No se encontró backend/static/index.html después del build"
  exit 1
fi

echo "[OK] Build listo"
