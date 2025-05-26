import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "..", "data", "annotations", "keypoints_dataset.csv")

df = pd.read_csv(CSV_PATH)
print(f"Dataset cargado: {df.shape[0]} muestras")

cols_vis = [col for col in df.columns if "_visibility" in col]
df = df.drop(columns=cols_vis)
varianzas = df.var(numeric_only=True)
cols_baja_varianza = varianzas[varianzas < 0.0001].index
df = df.drop(columns=cols_baja_varianza)
df = df.drop(columns=["frame"], errors="ignore")

features_angulos = [
    "angulo_codo_izquierdo", "angulo_rodilla_izquierda"
]
features_distancias = [
    "distancia_cadera_rodilla", "distancia_hombro_codo"
]

features_delta = features_angulos + features_distancias

for feat in features_delta:
    df[f"delta_{feat}"] = df.groupby("origen")[feat].diff().fillna(0)

X = df.drop(columns=["label", "origen"])
y = df["label"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

print("Entrenando modelo Logistic Regression con DELTAS...")
model = LogisticRegression(C=0.1, max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\nReporte de clasificación (con DELTAS):")
print(classification_report(y_test, y_pred))

from joblib import dump

EXPORT_DIR = os.path.join(SCRIPT_DIR, "..", "demo")
os.makedirs(EXPORT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(EXPORT_DIR, "logistic_pose_model.pkl")
SCALER_PATH = os.path.join(EXPORT_DIR, "scaler.pkl")

dump(model, MODEL_PATH)
dump(scaler, SCALER_PATH)

print(f"\nModelo exportado en: {MODEL_PATH}")
print(f"Scaler exportado en: {SCALER_PATH}")

FEATURES_PATH = os.path.join(EXPORT_DIR, "feature_names.txt")
with open(FEATURES_PATH, "w") as f:
    for col in X.columns:
        f.write(col + "\n")
print(f"Features exportadas en: {FEATURES_PATH}")