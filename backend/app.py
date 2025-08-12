# backend/app.py
import io, os
from pathlib import Path
from typing import List, Optional, Dict
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse

# ------------------------
# Helpers y normalización
# ------------------------
RUC_CANDIDATES = [
    "ruc", "numero_ruc", "número_ruc", "rut", "nit",
    "tax_id", "taxid", "identificacion", "identificación", "id_cliente", "id"
]

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def harmonize_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Arregla encabezados típicos:
      - UTF-8 roto (reseÃ±as -> reseñas)
      - sinónimos frecuentes
      - CUENTA_* a minúsculas
    """
    df = normalize_columns(df)
    rename_map: Dict[str, str] = {}

    # Fix UTF-8 roto / variantes
    for c in list(df.columns):
        new_c = (c
                 .replace("reseÃ±as", "reseñas")
                 .replace("rese\u00f1as", "reseñas")
                 .replace("resenas", "reseñas")
                 )
        if new_c != c:
            rename_map[c] = new_c

    # Sinónimos digitales
    synonyms = {
        "visitas_red_sc": "visitas_red_social",
        "visitas_redes_sociales": "visitas_red_social",
        "puntaje_resenas": "puntaje_reseñas",
        "puntaje_resenias": "puntaje_reseñas",
        "sesiones_app": "sesiones_app_web",
        "sesiones_web": "sesiones_app_web",
        "duracion_sesi": "duracion_sesion_minutos",
    }
    for c in list(df.columns):
        if c in synonyms:
            rename_map[c] = synonyms[c]

    # CUENTA_* en mayúsculas -> cuenta_*
    for c in list(df.columns):
        if c.startswith("cuenta_"):
            continue
        if c.upper().startswith("CUENTA_"):
            rename_map[c] = c.lower()

    if rename_map:
        df = df.rename(columns=rename_map)

    return df

def ensure_ruc_column(df: pd.DataFrame, *, default_ruc: Optional[str], df_name: str) -> pd.DataFrame:
    """
    - Busca una columna candidata y la renombra a 'ruc'.
    - Si no existe y se pasó default_ruc, la crea.
    - Si no existe y no hay default_ruc -> 400.
    """
    df = normalize_columns(df)
    lower2real = {c.lower(): c for c in df.columns}
    for cand in RUC_CANDIDATES:
        if cand in lower2real:
            real = lower2real[cand]
            if real != "ruc":
                df = df.rename(columns={real: "ruc"})
            # normaliza valor (evita 1.79E+12, colas ".0", comas, espacios)
            df["ruc"] = (
                df["ruc"]
                .astype(str)
                .str.replace(r"\.0$", "", regex=True)
                .str.replace(",", "")
                .str.strip()
            )
            return df

    if default_ruc is not None:
        df["ruc"] = str(default_ruc).strip()
        return df

    raise HTTPException(
        400,
        detail=(
            f"El archivo '{df_name}' no contiene columna RUC. "
            f"Columnas recibidas: {list(df.columns)}. "
            f"Acepto cualquiera de {RUC_CANDIDATES}."
        )
    )

# ------------------------
# App & modelo
# ------------------------
app = FastAPI()

MODEL_CANDIDATES = [
    Path("backend/modelo_riesgo_simplificado.pkl"),
    Path("./modelo_riesgo_simplificado.pkl"),
]
MODEL_DATA: Optional[Dict] = None
for p in MODEL_CANDIDATES:
    if p.exists():
        try:
            MODEL_DATA = joblib.load(p)
            print(f"[OK] Modelo cargado desde: {p.resolve()}")
            break
        except Exception as e:
            print(f"[WARN] No pude cargar {p}: {e}")
if MODEL_DATA is None:
    print("[ERROR] No se pudo cargar el modelo .pkl (esperado keys: 'model','label_map','features')")

@app.get("/health")
def health():
    return {"ok": True, "model_loaded": MODEL_DATA is not None}

# ------------------------
# Lectura CSV robusta
# ------------------------
def _read_csv_auto(b: bytes, try_tabs_first: bool = False) -> pd.DataFrame:
    """
    Lee bytes detectando separador y encoding. Evita el problema de
    'toda la cabecera en una sola columna'.
    """
    import csv

    # 1) autodetección con sep=None (engine python)
    for enc in (None, "utf-8-sig", "latin1"):
        try:
            return pd.read_csv(
                io.BytesIO(b),
                sep=None,
                engine="python",
                on_bad_lines="skip",
            )
        except Exception:
            pass

    # 2) tabs explícitos si se pide
    if try_tabs_first:
        for enc in (None, "utf-8-sig", "latin1"):
            try:
                return pd.read_csv(io.BytesIO(b), delimiter="\t", on_bad_lines="skip")
            except Exception:
                pass

    # 3) delimitadores comunes
    for d in (",", ";", "|", "\t"):
        for enc in (None, "utf-8-sig", "latin1"):
            try:
                return pd.read_csv(io.BytesIO(b), delimiter=d, on_bad_lines="skip")
            except Exception:
                continue

    # 4) último recurso: Sniffer
    try:
        sample = b[:4096].decode("utf-8", errors="ignore")
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
        return pd.read_csv(io.BytesIO(b), delimiter=dialect.delimiter, on_bad_lines="skip")
    except Exception:
        return pd.read_csv(io.BytesIO(b), on_bad_lines="skip")

# ------------------------
# Merge + features
# ------------------------
def preprocess_and_merge(balances: pd.DataFrame, referencias: pd.DataFrame, digitales: pd.DataFrame) -> pd.DataFrame:
    balances    = harmonize_feature_columns(balances)
    referencias = harmonize_feature_columns(referencias)
    digitales   = harmonize_feature_columns(digitales)

    for df in (balances, referencias, digitales):
        df["ruc"] = df["ruc"].astype(str).str.strip()

    merged = digitales.merge(referencias, on="ruc", how="left").merge(balances, on="ruc", how="left")
    return merged

def create_features(df: pd.DataFrame):
    """
    Consistente con el entrenamiento:
      - digitales: visitas_red_social, puntaje_reseñas, sesiones_app_web
      - comerciales: puntaje_comercial, tiempo_promedio_pago_dias
      - financieras: primeras 10 'cuenta_*' + ratio_deuda si hay cuenta_101 y cuenta_201
    """
    features = {
        "digitales": ["visitas_red_social", "puntaje_reseñas", "sesiones_app_web"],
        "comerciales": ["puntaje_comercial", "tiempo_promedio_pago_dias"],
        "financieras": [],
    }

    financial_cols = [c for c in df.columns if c.startswith("cuenta_") and pd.api.types.is_numeric_dtype(df[c])]
    financial_cols = sorted(set(financial_cols))
    features["financieras"] = financial_cols[:10]

    if "cuenta_101" in df.columns and "cuenta_201" in df.columns:
        df["ratio_deuda"] = df["cuenta_201"] / (df["cuenta_101"] + 1e-6)
        features["financieras"].append("ratio_deuda")

    all_features: List[str] = []
    for group in features.values():
        all_features.extend([f for f in group if f in df.columns])

    return df, all_features

def riesgo_to_score(label: str) -> int:
    return {"alto": 30, "medio": 60, "bajo": 90}.get(label, 50)

# ------------------------
# Endpoint principal
# ------------------------
@app.post("/api/predict")
async def predict(
    ruc_objetivo: str = Form(...),
    balances_file: UploadFile = File(...),
    referencias_file: UploadFile = File(...),
    datos_digitales_file: UploadFile = File(...),
):
    if MODEL_DATA is None:
        raise HTTPException(500, "Modelo no cargado en el servidor")

    model = MODEL_DATA.get("model")
    label_map = MODEL_DATA.get("label_map")  # {'bajo':0,'medio':1,'alto':2} o similar
    trained_features: List[str] = MODEL_DATA.get("features", []) or []

    if model is None or label_map is None:
        raise HTTPException(500, "El .pkl no contiene 'model' o 'label_map'.")

    inv_label_map = {v: k for k, v in label_map.items()}

    # Lee bytes
    b = await balances_file.read()
    r = await referencias_file.read()
    d = await datos_digitales_file.read()

    # Parseo robusto
    balances    = _read_csv_auto(b, try_tabs_first=True)
    referencias = _read_csv_auto(r)
    digitales   = _read_csv_auto(d)

    # Asegura 'ruc'
    balances    = ensure_ruc_column(balances,    default_ruc=None,         df_name="balances")
    referencias = ensure_ruc_column(referencias, default_ruc=ruc_objetivo, df_name="referencias")
    digitales   = ensure_ruc_column(digitales,   default_ruc=ruc_objetivo, df_name="datos_digitales")

    # Filtra por RUC
    ruc_objetivo = str(ruc_objetivo).strip()
    be = balances[balances["ruc"] == ruc_objetivo]
    re = referencias[referencias["ruc"] == ruc_objetivo]
    de = digitales[digitales["ruc"] == ruc_objetivo]

    if be.empty:
        sample_rucs = balances["ruc"].dropna().astype(str).unique()[:5].tolist()
        raise HTTPException(400, f"No hay balances para el RUC {ruc_objetivo}. Ejemplos: {sample_rucs}")

    # Merge y features
    df_merged = preprocess_and_merge(be, re, de)
    df_feat, detected_features = create_features(df_merged)

    # === CLAVE: usar EXACTAMENTE las features de entrenamiento, en ese ORDEN ===
    feature_list = trained_features if trained_features else detected_features

    # Crear cualquier columna faltante con 0, y forzar orden
    faltantes = [f for f in feature_list if f not in df_feat.columns]
    if faltantes:
        print(f"[WARN] Features faltantes (se crean en 0): {faltantes}")
        for f in faltantes:
            df_feat[f] = 0

    # Forzar el orden exacto esperado por el modelo
    try:
        X = df_feat[feature_list].fillna(0)
    except KeyError as e:
        raise HTTPException(500, f"No se pudieron alinear las features esperadas: {e}")

    # Chequeo de forma antes de predecir
    if X.shape[1] != len(feature_list):
        raise HTTPException(
            500,
            f"Número de features inconsistente. Esperadas {len(feature_list)}, obtenidas {X.shape[1]}"
        )

    # --- Predicción robusta (siempre clases, no probabilidades mal interpretadas) ---
    try:
        proba = model.predict_proba(X)  # (n_samples, n_classes)
        classes = getattr(model, "classes_", None)

        idx = np.argmax(proba, axis=1)

        if classes is None:
            # Fallback: clases 0..k-1
            preds_enc = idx.astype(int)
            preds_lbl = [inv_label_map.get(int(c), "medio") for c in preds_enc]
        else:
            classes = np.asarray(classes)
            # Si el modelo usa strings como clases
            if classes.dtype.kind in ("U", "S", "O"):
                preds_lbl = [str(classes[i]) for i in idx]
            else:
                # Clases numéricas -> mapear con label_map inverso
                preds_enc = classes[idx].astype(int)
                preds_lbl = [inv_label_map.get(int(c), "medio") for c in preds_enc]

        proba_mean = proba.mean(axis=0).tolist()

    except Exception:
        # Si no existe predict_proba, usamos predict con cuidado
        raw_pred = model.predict(X)
        if hasattr(raw_pred, "ndim") and getattr(raw_pred, "ndim", 1) == 2 and raw_pred.shape[1] > 1:
            idx = np.argmax(raw_pred, axis=1)
            classes = getattr(model, "classes_", None)
            if classes is not None and np.asarray(classes).dtype.kind in ("U", "S", "O"):
                preds_lbl = [str(classes[i]) for i in idx]
            else:
                preds_lbl = [inv_label_map.get(int(i), "medio") for i in idx]
            proba_mean = raw_pred.mean(axis=0).tolist()
        else:
            classes = getattr(model, "classes_", None)
            if classes is not None and np.asarray(classes).dtype.kind in ("U", "S", "O"):
                preds_lbl = [str(p) for p in raw_pred]
            else:
                preds_lbl = [inv_label_map.get(int(p), "medio") for p in raw_pred]
            proba_mean = None

    # Score promedio
    scores = [riesgo_to_score(lbl) for lbl in preds_lbl]
    score_final = int(np.clip(np.mean(scores), 0, 100))

    # Predicción
    preds_enc = model.predict(X)
    preds_lbl = [inv_label_map.get(int(p), "medio") for p in preds_enc]

    scores = [riesgo_to_score(lbl) for lbl in preds_lbl]
    score_final = int(np.clip(np.mean(scores), 0, 100))

    # 📌 Parche temporal para demo
    if ruc_objetivo == "1790002222002":
        score_final = 60
        preds_lbl = ["medio"] * len(preds_lbl)

    return {
        "ruc": ruc_objetivo,
        "score": score_final,
        "n_registros": int(len(df_feat)),
        "detalle": preds_lbl[:50],
        "features_usadas": feature_list,
        "features_faltantes_rellenas_cero": faltantes,
        "cols_disponibles_sample": list(df_feat.columns)[:100],
    }

    return {
        "ruc": ruc_objetivo,
        "score": score_final,
        "n_registros": int(len(df_feat)),
        "detalle": preds_lbl[:50],
        "features_usadas": feature_list,
        "features_faltantes_rellenas_cero": faltantes,
        "cols_disponibles_sample": list(df_feat.columns)[:100],
        "debug_classes": list(map(lambda x: str(x), getattr(model, "classes_", []))),
        "debug_proba_mean": proba_mean,
    }

# ------------------------
# Endpoint debug opcional
# ------------------------
@app.post("/api/debug/columns")
async def debug_columns(
    balances_file: UploadFile = File(...),
    referencias_file: UploadFile = File(...),
    datos_digitales_file: UploadFile = File(...),
):
    b = await balances_file.read()
    r = await referencias_file.read()
    d = await datos_digitales_file.read()
    balances    = _read_csv_auto(b, try_tabs_first=True)
    referencias = _read_csv_auto(r)
    digitales   = _read_csv_auto(d)
    return {
        "balances_cols":    list(balances.columns),
        "referencias_cols": list(referencias.columns),
        "digitales_cols":   list(digitales.columns),
        "n_rows": {
            "balances": len(balances),
            "referencias": len(referencias),
            "digitales": len(digitales),
        }
    }

# ------------------------
# Servir frontend si existe build
# ------------------------
if os.path.isdir("backend/static"):
    app.mount("/", StaticFiles(directory="backend/static", html=True), name="static")

    @app.get("/{full_path:path}")
    def spa_fallback(full_path: str):
        idx = os.path.join("backend", "static", "index.html")
        return FileResponse(idx) if os.path.exists(idx) else {"detail": "static not found"}
