import os
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from joblib import dump

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "..", "data", "annotations", "keypoints_dataset_clean.csv")
EXPORT_DIR = os.path.join(SCRIPT_DIR, "..", "demo")
MODEL_PATH = os.path.join(EXPORT_DIR, "gradientboost_pose_model.pkl")
FEATURES_PATH = os.path.join(EXPORT_DIR, "feature_names.txt")
METRICS_PATH = os.path.join(EXPORT_DIR, "metrics.txt")


def cargar_dataset(path):
    df = pd.read_csv(path)
    logger.info(f"Dataset cargado: {df.shape[0]} muestras, {df.shape[1]} columnas")
    return df

def limpiar_datos(df):
    cols_vis = [col for col in df.columns if "_visibility" in col]
    df.drop(columns=cols_vis, inplace=True)

    varianzas = df.var(numeric_only=True)
    cols_baja_varianza = varianzas[varianzas < 0.0001].index
    df.drop(columns=cols_baja_varianza, inplace=True)

    df.drop(columns=["frame"], errors="ignore", inplace=True)
    return df


def crear_deltas(df, features, group_col="origen"):
    for feat in features:
        if feat in df.columns:
            df[f"delta_{feat}"] = df.groupby(group_col)[feat].diff().fillna(0)
        else:
            logger.warning(f"Feature {feat} no encontrada para crear delta.")
    return df

def validar_columnas(df, columnas_requeridas):
    columnas_presentes = [col for col in columnas_requeridas if col in df.columns]
    faltantes = set(columnas_requeridas) - set(columnas_presentes)
    if faltantes:
        logger.warning(f"Columnas faltantes que no se usarán: {faltantes}")
    return columnas_presentes


def filtrar_clases_insuficientes(df, min_muestras=2):
    counts = df["label"].value_counts()
    clases_validas = counts[counts >= min_muestras].index
    df_filtrado = df[df["label"].isin(clases_validas)]
    logger.info(f"Clases válidas después del filtro: {list(clases_validas)}")
    return df_filtrado


def guardar_resultados(modelo, columnas, reporte):
    os.makedirs(EXPORT_DIR, exist_ok=True)
    dump(modelo, MODEL_PATH)
    with open(FEATURES_PATH, "w") as f:
        for col in columnas:
            f.write(col + "\n")
    with open(METRICS_PATH, "w") as f:
        f.write(reporte)
    logger.info(f"Modelo exportado en: {MODEL_PATH}")
    logger.info(f"Features exportadas en: {FEATURES_PATH}")
    logger.info(f"Métricas exportadas en: {METRICS_PATH}")


def main():
    df = cargar_dataset(CSV_PATH)
    df = limpiar_datos(df)

    columnas_requeridas = [
        "distancia_cadera_rodilla",
        "distancia_hombro_codo",
        "angulo_codo_izquierdo",
        "angulo_rodilla_izquierda"
    ]
    features_delta = validar_columnas(df, columnas_requeridas)
    df = crear_deltas(df, features_delta)

    if "label" not in df.columns:
        raise ValueError("La columna 'label' no está presente en el dataset.")
    if "origen" not in df.columns:
        raise ValueError("La columna 'origen' no está presente en el dataset.")

    df = filtrar_clases_insuficientes(df, min_muestras=2)

    X = df.drop(columns=["label", "origen"])
    y = df["label"]

    if X.isnull().sum().sum() > 0:
        logger.warning("Hay valores nulos en los datos. Se recomienda imputarlos o eliminarlos.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("gb", GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
        ))
    ])

    logger.info("Entrenando modelo Gradient Boosting con DELTAS...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    reporte = classification_report(y_test, y_pred)
    print("\nReporte de clasificación:\n", reporte)

    guardar_resultados(pipeline, X.columns, reporte)

if __name__ == "__main__":
    main()