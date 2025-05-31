import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from joblib import dump

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "..", "data", "annotations", "keypoints_dataset_clean.csv")

df = pd.read_csv(CSV_PATH)
print(f"Dataset cargado: {df.shape[0]} muestras")

cols_vis = [col for col in df.columns if "_visibility" in col]
df.drop(columns=cols_vis, inplace=True)

varianzas = df.var(numeric_only=True)
cols_baja_varianza = varianzas[varianzas < 0.0001].index
df.drop(columns=cols_baja_varianza, inplace=True)
df.drop(columns=["frame"], errors="ignore", inplace=True)

features_angulos = ["angulo_codo_izquierdo", "angulo_rodilla_izquierda"]
features_distancias = ["distancia_cadera_rodilla", "distancia_hombro_codo"]
features_delta = features_angulos + features_distancias

required_cols = [
    col for col in [
        "distancia_cadera_rodilla",
        "distancia_hombro_codo",
        "angulo_codo_izquierdo",
        "angulo_rodilla_izquierda"
    ] if col in df.columns
]

missing = set(["distancia_cadera_rodilla", "distancia_hombro_codo",
               "angulo_codo_izquierdo", "angulo_rodilla_izquierda"]) - set(required_cols)
if missing:
    print(f"Algunas columnas no están presentes y no se usarán: {missing}")


features_delta = required_cols

for feat in features_delta:
    df[f"delta_{feat}"] = df.groupby("origen")[feat].diff().fillna(0)

X = df.drop(columns=["label", "origen"])
y = df["label"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

print("Entrenando modelo Gradient Boosting con DELTAS...")
model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\nReporte de clasificación (con DELTAS):")
print(classification_report(y_test, y_pred))

EXPORT_DIR = os.path.join(SCRIPT_DIR, "..", "demo")
os.makedirs(EXPORT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(EXPORT_DIR, "gradientboost_pose_model.pkl")
SCALER_PATH = os.path.join(EXPORT_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(EXPORT_DIR, "feature_names.txt")

dump(model, MODEL_PATH)
dump(scaler, SCALER_PATH)

with open(FEATURES_PATH, "w") as f:
    for col in X.columns:
        f.write(col + "\n")

print(f"\nModelo exportado en: {MODEL_PATH}")
print(f"Scaler exportado en: {SCALER_PATH}")
print(f"Features exportadas en: {FEATURES_PATH}")