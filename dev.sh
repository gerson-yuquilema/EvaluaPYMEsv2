#!/bin/bash
set -Eeuo pipefail

# Mata hijos al salir
cleanup() {
  echo "Shutting down dev services..."
  pkill -P $$ || true
}
trap cleanup EXIT INT TERM

# --- Backend ---
echo "===> Starting backend"
cd backend
python -m venv venv || true
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Arranca backend en background
( uvicorn app:app --reload --host 127.0.0.1 --port 8000 ) &
BACK_PID=$!

# Espera health
echo "===> Waiting backend health"
for i in {1..40}; do
  if curl -sSf http://127.0.0.1:8000/health >/dev/null; then
    echo "[OK] Backend up"
    break
  fi
  sleep 0.5
  if ! kill -0 $BACK_PID 2>/dev/null; then
    echo "[ERROR] Backend process died early"
    exit 1
  fi
done

cd ..

# --- Frontend ---
echo "===> Starting frontend"
cd frontend
npm install
# Vite tomará el proxy a 127.0.0.1:8000
npm run dev
