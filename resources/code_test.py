import pandas as pd
import numpy as np
import joblib
import os

def load_data():
    """Carga los tres conjuntos de datos"""
    folder_path = os.path.dirname(os.path.abspath(_file_))
    
    digitales = pd.read_csv(os.path.join(folder_path, 'simulacion_datos_digitales.csv'))
    referencias = pd.read_csv(os.path.join(folder_path, 'simulacion_referencias_comerciales.csv'))
    try:
        balances = pd.read_csv(os.path.join(folder_path, 'simulacion_balances_completo.txt'), 
                               delimiter='\t', encoding='utf-8')
    except:
        try:
            balances = pd.read_csv(os.path.join(folder_path, 'simulacion_balances_completo.txt'), 
                                   delimiter='\t', encoding='latin1')
        except:
            balances = pd.read_csv(os.path.join(folder_path, 'simulacion_balances_completo.txt'), 
                                   delimiter='\t', encoding='ISO-8859-1')
    return balances, referencias, digitales

def preprocess_and_merge(balances, referencias, digitales):
    """Preprocesa y combina los datasets"""
    for df in [balances, referencias, digitales]:
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
    financial_cols = [col for col in df.columns if col.startswith('cuenta_') and pd.api.types.is_numeric_dtype(df[col])]
    features['financieras'] = financial_cols[:10]
    
    try:
        if 'cuenta_101' in df.columns and 'cuenta_201' in df.columns:
            df['ratio_deuda'] = df['cuenta_201'] / (df['cuenta_101'] + 1e-6)
            features['financieras'].append('ratio_deuda')
    except:
        pass
    
    all_features = []
    for category in features.values():
        all_features.extend([f for f in category if f in df.columns])
    
    return df, all_features

def predict_risk_for_ruc(ruc_input):
    print("📦 Cargando modelo entrenado...")
    model_data = joblib.load('modelo_riesgo_simplificado.pkl')
    model = model_data['model']
    label_map = model_data['label_map']
    features = model_data['features']
    
    print("⏳ Cargando datos...")
    balances, referencias, digitales = load_data()
    
    print("🔧 Procesando datos...")
    df = preprocess_and_merge(balances, referencias, digitales)
    df, _ = create_features(df)
    
    # Filtrar por RUC
    df_ruc = df[df['ruc'] == ruc_input].copy()
    if df_ruc.empty:
        print(f"No se encontraron datos para el RUC {ruc_input}")
        return
    
    print(f"🔮 Realizando predicciones para RUC {ruc_input}...")
    
    X_pred = df_ruc[features].fillna(0)
    preds_encoded = model.predict(X_pred)
    
    # Mapear etiquetas de riesgo
    inv_label_map = {v: k for k, v in label_map.items()}
    preds_labels = [inv_label_map[p] for p in preds_encoded]
    
    df_ruc['riesgo_predicho'] = preds_labels
    
    print("\nResultados de predicción (RUC y riesgo):")
    print(df_ruc[['ruc', 'riesgo_predicho']].drop_duplicates())
    
    # Agrupar por RUC y obtener riesgo máximo
    risk_order = {'bajo': 0, 'medio': 1, 'alto': 2}
    df_ruc['risk_score'] = df_ruc['riesgo_predicho'].map(risk_order)
    
    max_risk_score = df_ruc['risk_score'].max()
    max_risk_label = [k for k, v in risk_order.items() if v == max_risk_score][0]
    
    print(f"\nRiesgo máximo predicho para el RUC {ruc_input}: {max_risk_label}")

if _name_ == "_main_":
    ruc_input = input("Ingresa el RUC para predecir riesgo: ").strip()
    predict_risk_for_ruc(ruc_input)