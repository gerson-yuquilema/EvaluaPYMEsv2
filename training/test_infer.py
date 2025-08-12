# training/test_infer.py
import os, joblib
import pandas as pd
from train_risk_model import preprocess_and_merge, create_features, load_data

def predict_risk_for_ruc(ruc_input: str):
    model_path = os.path.join(os.path.dirname(__file__), "..", "backend", "modelo_riesgo_simplificado.pkl")
    md = joblib.load(model_path)
    model, label_map, features = md['model'], md['label_map'], md['features']

    folder_path = os.path.dirname(os.path.abspath(__file__))
    balances, referencias, digitales = load_data(folder_path)
    df = preprocess_and_merge(balances, referencias, digitales)
    df, _ = create_features(df)

    df_ruc = df[df['ruc'].astype(str).str.strip() == str(ruc_input).strip()]
    if df_ruc.empty:
        print(f"No se encontraron datos para el RUC {ruc_input}")
        return

    X_pred = df_ruc[features].fillna(0)
    preds_encoded = model.predict(X_pred)
    inv = {v:k for k,v in label_map.items()}
    preds_labels = [inv[p] for p in preds_encoded]
    print(df_ruc[['ruc']].drop_duplicates().assign(riesgo_predicho=preds_labels[0]))

if __name__ == "__main__":
    ruc = input("Ingresa el RUC: ").strip()
    predict_risk_for_ruc(ruc)
