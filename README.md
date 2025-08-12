# Evaluapy Score (PYME Risk Scoring)

Evaluapy es una app que genera un **score de riesgo crediticio** (0–100) para PYMEs a partir de:
- **Balances** (cuentas contables mínimas)
- **Referencias comerciales**
- **Señales digitales** (actividad/redes)

El **backend** está hecho con **FastAPI** y sirve también el **frontend** (React) precompilado dentro de `backend/static` (SPA).  
El modelo se carga desde `modelo_riesgo_simplificado.pkl` (LightGBM u otro sklearn-compatible) y mapea las clases a un score:
- **bajo → 90**
- **medio → 60**
- **alto → 30**

> Nota: hay un *parche temporal* para el RUC `1790002222002` que fuerza **riesgo medio** (60), útil para demo.

---

## Estructura del repo

├─ backend/
│ ├─ app.py
│ ├─ requirements.txt
│ ├─ modelo_riesgo_simplificado.pkl
│ └─ static/ # (se llena al hacer build del frontend)
├─ frontend/
│ └─ ... (React/Vite)
├─ resources/
│ ├─ README_TEST_DATA.txt # guía de RUCs de prueba
│ ├─ rucs_prueba.txt # lista de RUCs de ejemplo
│ ├─ balances_demo.csv # balances de prueba (mínimos)
│ ├─ referencias_demo.csv # referencias comerciales de prueba
│ └─ digitales_demo.csv # señales digitales de prueba
├─ build.sh # compila el frontend y lo copia a backend/static
├─ dev.sh # levanta backend y frontend en modo desarrollo
└─ README.md # este archivo

---

## Cómo funciona (alto nivel)

1. El frontend guía por pasos para subir **3 archivos**:
   - **Balances** (`balances_demo.csv` o `.txt`)
   - **Referencias** (`referencias_demo.csv`)
   - **Digitales** (`digitales_demo.csv`)
2. El backend armoniza encabezados, asegura `ruc` y hace **merge** por `ruc`.
3. Se construye el vector de **features** y se pasa al modelo.
4. El modelo predice clase (`alto`/`medio`/`bajo`) y se convierte a score.
5. El frontend muestra el resultado y permite descargar un PDF.

---

## Requisitos

- Python **3.12**
- Node 18+ (solo si vas a recompilar el frontend)

---

## Ejecución en entorno local

### Modo desarrollo (recomendado para pruebas)
```bash
chmod +x dev.sh
./dev.sh

Esto levanta backend y frontend en modo hot-reload.

# Crear venv e instalar deps
python3.12 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt

# (Opcional) Construir frontend
bash build.sh

# Levantar backend
cd backend
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

Data de prueba (resources/)
Escenario	RUC	Esperado
Alto	1790001111001	~30
Medio	1790002222002	~60 (*)
Bajo	1790003333003	~90

(*) Parche demo activo.

Archivos:

balances_demo.csv

referencias_demo.csv

digitales_demo.csv

Probar API con curl

curl -X POST http://localhost:8000/api/predict \
  -F ruc_objetivo=1790002222002 \
  -F balances_file=@resources/balances_demo.csv \
  -F referencias_file=@resources/referencias_demo.csv \
  -F datos_digitales_file=@resources/digitales_demo.csv

Despliegue en Render
Sube el repo a GitHub.

En Render → New → Web Service → conecta repo.

Configura:

Build Command: bash build.sh

Start Command: cd backend && uvicorn app:app --host 0.0.0.0 --port $PORT

Plan: Starter para 24/7.

Deploy.

Notas
El backend corrige encabezados y notación científica de RUC.

Usa exactamente las features entrenadas.

El parche para 1790002222002 está en app.py.# EvaluaPYMEs
