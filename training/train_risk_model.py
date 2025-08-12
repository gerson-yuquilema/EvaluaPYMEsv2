# training/train_risk_model.py
import os, joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def load_data(folder_path: str):
    digitales = pd.read_csv(os.path.join(folder_path, 'simulacion_datos_digitales.csv'))
    referencias = pd.read_csv(os.path.join(folder_path, 'simulacion_referencias_comerciales.csv'))
    balances = None
    for enc in ('utf-8', 'latin1', 'ISO-8859-1'):
        try:
            balances = pd.read_csv(
                os.path.join(folder_path, 'simulacion_balances_completo.txt'),
                delimiter='\t', encoding=enc
            )
            break
        except Exception:
            continue
    if balances is None:
        raise RuntimeError("No pude leer simulacion_balances_completo.txt con utf-8/latin1/ISO-8859-1")
    return balances, referencias, digitales

def preprocess_and_merge(balances, referencias, digitales):
    for df in (balances, referencias, digitales):
        df.columns = df.columns.str.strip().str.lower()
        df['ruc'] = df['ruc'].astype(str).str.strip()
    merged = digitales.merge(referencias, on='ruc', how='left').merge(balances, on='ruc', how='left')
    numeric_cols = balances.select_dtypes(include=['number']).columns.tolist()
    if 'ruc' in numeric_cols:
        numeric_cols.remove('ruc')
    for col in numeric_cols:
        if col in balances.columns:
            merged[col] = balances[col]
    return merged

def create_features(df):
    features = {
        'digitales': ['visitas_red_social', 'puntaje_reseñas', 'sesiones_app_web'],
        'comerciales': ['puntaje_comercial', 'tiempo_promedio_pago_dias'],
        'financieras': []
    }
    financial_cols = [c for c in df.columns if c.startswith('cuenta_') and pd.api.types.is_numeric_dtype(df[c])]
    features['financieras'] = financial_cols[:10]
    if 'cuenta_101' in df.columns and 'cuenta_201' in df.columns:
        df['ratio_deuda'] = df['cuenta_201'] / (df['cuenta_101'] + 1e-6)
        features['financieras'].append('ratio_deuda')
    all_features = []
    for group in features.values():
        all_features.extend([f for f in group if f in df.columns])
    return df, all_features

def generate_labels(df):
    factors = []
    if all(c in df.columns for c in ['visitas_red_social', 'puntaje_reseñas', 'sesiones_app_web']):
        df['digital_score'] = (
            df['visitas_red_social'] * 0.3 +
            df['puntaje_reseñas'] * 0.5 +
            df['sesiones_app_web'] * 0.2
        )
        factors.append('digital_score')
    if all(c in df.columns for c in ['puntaje_comercial', 'tiempo_promedio_pago_dias']):
        df['commercial_score'] = (
            df['puntaje_comercial'] * 0.6 +
            (60 - df['tiempo_promedio_pago_dias']).clip(lower=0) / 60 * 0.4
        )
        factors.append('commercial_score')
    financial_cols = [c for c in df.columns if c.startswith('cuenta_') and pd.api.types.is_numeric_dtype(df[c])]
    if financial_cols:
        df['financial_score'] = df[financial_cols[0]]
        factors.append('financial_score')
    if factors:
        df['total_score'] = df[factors].mean(axis=1)
        df['riesgo'] = pd.qcut(df['total_score'], q=3, labels=['alto', 'medio', 'bajo'])
    else:
        df['riesgo'] = 'medio'
    return df

def train_model(df, features):
    X = df[features].fillna(0)
    y = df['riesgo']
    label_map = {'bajo': 0, 'medio': 1, 'alto': 2}
    y_encoded = y.map(label_map)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    model = lgb.LGBMClassifier(objective='multiclass', num_class=3, random_state=42, verbosity=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred, target_names=list(label_map.keys())))
    return model, label_map

def main():
    folder_path = os.path.dirname(os.path.abspath(__file__))
    print("📥 Cargando datos…")
    balances, referencias, digitales = load_data(folder_path)
    print("🧹 Procesando…")
    df = preprocess_and_merge(balances, referencias, digitales)
    df, features = create_features(df)
    df = generate_labels(df)
    print("🤖 Entrenando…")
    model, label_map = train_model(df, features)
    out_path = os.path.join(os.path.dirname(folder_path), "backend", "modelo_riesgo_simplificado.pkl")
    joblib.dump({'model': model, 'label_map': label_map, 'features': features}, out_path)
    print(f"✅ Modelo guardado en: {out_path}")
    print(f"🔢 {len(features)} features usadas.")

if __name__ == "__main__":
    main()
