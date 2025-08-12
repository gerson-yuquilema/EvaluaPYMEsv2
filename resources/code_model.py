import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

def load_data():
    """Carga los tres conjuntos de datos"""
    folder_path = os.path.dirname(os.path.abspath(_file_))
    
    # Cargar datos digitales
    digitales = pd.read_csv(os.path.join(folder_path, 'simulacion_datos_digitales.csv'))
    
    # Cargar referencias comerciales
    referencias = pd.read_csv(os.path.join(folder_path, 'simulacion_referencias_comerciales.csv'))
    
    # Cargar balances (manejar diferentes formatos)
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
    # Limpieza básica
    for df in [balances, referencias, digitales]:
        df.columns = df.columns.str.strip().str.lower()
        df['ruc'] = df['ruc'].astype(str).str.strip()
    
    # Fusionar los datasets
    merged = digitales.merge(referencias, on='ruc', how='left').merge(balances, on='ruc', how='left')
    
    # Seleccionar solo columnas numéricas de balances (excluyendo columnas de texto)
    numeric_cols = balances.select_dtypes(include=['number']).columns.tolist()
    if 'ruc' in numeric_cols:
        numeric_cols.remove('ruc')
    
    # Agregar columnas seleccionadas de balances
    for col in numeric_cols:
        if col in balances.columns:
            merged[col] = balances[col]
    
    return merged

def create_features(df):
    """Crea características para el modelo"""
    # Características básicas (asegurarse de que existen)
    features = {
        'digitales': ['visitas_red_social', 'puntaje_reseñas', 'sesiones_app_web'],
        'comerciales': ['puntaje_comercial', 'tiempo_promedio_pago_dias'],
        'financieras': []  # Se llenará dinámicamente
    }
    
    # Agregar columnas financieras numéricas (si existen)
    financial_cols = [col for col in df.columns if col.startswith('cuenta_') and pd.api.types.is_numeric_dtype(df[col])]
    features['financieras'] = financial_cols[:10]  # Limitar a 10 columnas financieras para simplificar
    
    # Crear características compuestas
    try:
        if 'cuenta_101' in df.columns and 'cuenta_201' in df.columns:
            df['ratio_deuda'] = df['cuenta_201'] / (df['cuenta_101'] + 1e-6)
            features['financieras'].append('ratio_deuda')
    except:
        pass
    
    # Seleccionar todas las características disponibles
    all_features = []
    for category in features.values():
        all_features.extend([f for f in category if f in df.columns])
    
    return df, all_features

def generate_labels(df):
    """Genera etiquetas de riesgo basado en características disponibles"""
    # Factores con pesos
    factors = []
    
    # 1. Factor digital (si las columnas existen)
    digital_cols = ['visitas_red_social', 'puntaje_reseñas', 'sesiones_app_web']
    if all(col in df.columns for col in digital_cols):
        df['digital_score'] = (df['visitas_red_social'] * 0.3 + 
                             df['puntaje_reseñas'] * 0.5 + 
                             df['sesiones_app_web'] * 0.2)
        factors.append('digital_score')
    
    # 2. Factor comercial
    commercial_cols = ['puntaje_comercial', 'tiempo_promedio_pago_dias']
    if all(col in df.columns for col in commercial_cols):
        df['commercial_score'] = (df['puntaje_comercial'] * 0.6 + 
                                (60 - df['tiempo_promedio_pago_dias']) / 60 * 0.4)
        factors.append('commercial_score')
    
    # 3. Factor financiero (usar primera columna financiera disponible)
    financial_cols = [col for col in df.columns if col.startswith('cuenta_')]
    if financial_cols:
        df['financial_score'] = df[financial_cols[0]]  # Usar la primera columna financiera
        factors.append('financial_score')
    
    # Crear score combinado (si hay factores)
    if factors:
        df['total_score'] = df[factors].mean(axis=1)
        
        # Clasificar en categorías de riesgo
        df['riesgo'] = pd.qcut(df['total_score'], q=3, labels=['alto', 'medio', 'bajo'])
    else:
        # Si no hay factores, asignar riesgo medio a todos
        df['riesgo'] = 'medio'
    
    return df

def train_model(df, features):
    """Entrena el modelo LightGBM"""
    # Preparar datos
    X = df[features].fillna(0)
    y = df['riesgo']
    
    # Codificar etiquetas
    label_map = {'bajo': 0, 'medio': 1, 'alto': 2}
    y_encoded = y.map(label_map)
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Entrenar modelo
    model = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=3,
        random_state=42,
        verbosity=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluar
    y_pred = model.predict(X_test)
    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred, 
                              target_names=label_map.keys()))
    
    return model, label_map

def main():
    print(" Cargando datos...")
    balances, referencias, digitales = load_data()
    
    print("🔧 Procesando datos...")
    df = preprocess_and_merge(balances, referencias, digitales)
    df, features = create_features(df)
    df = generate_labels(df)
    
    print("🤖 Entrenando modelo...")
    model, label_map = train_model(df, features)
    
    print("💾 Guardando modelo...")
    joblib.dump({
        'model': model,
        'label_map': label_map,
        'features': features
    }, 'modelo_riesgo_simplificado.pkl')
    
    print(f" Modelo entrenado con {len(features)} características y guardado")

if _name_ == "_main_":
    main()