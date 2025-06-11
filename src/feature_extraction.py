import os
import json
import pandas as pd
import numpy as np
from math import acos, degrees

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_DIR = os.path.join(SCRIPT_DIR, "..", "data", "processed")
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "..", "data", "annotations", "keypoints_dataset.csv")

def json_to_dataframe(json_path, origen):
    with open(json_path) as f:
        data = json.load(f)

    rows = []
    for entry in data:
        row = {
            'frame': entry.get('frame', -1),
            'time_seconds': entry.get('time_seconds', None),
            'label': entry.get('label', 'unknown'),
            'origen': origen
        }
        for point_id, values in entry.get('keypoints', {}).items():
            for coord, value in values.items():
                row[f"{point_id}_{coord}"] = value
        rows.append(row)

    return pd.DataFrame(rows)

def distancia(df, p1, p2):
    return np.sqrt((df[f"{p1}_x"] - df[f"{p2}_x"])**2 + (df[f"{p1}_y"] - df[f"{p2}_y"])**2)

def angulo(df, a, b, c):
    vec_ab = np.stack([
        df[f"{a}_x"] - df[f"{b}_x"],
        df[f"{a}_y"] - df[f"{b}_y"]
    ], axis=1)
    vec_cb = np.stack([
        df[f"{c}_x"] - df[f"{b}_x"],
        df[f"{c}_y"] - df[f"{b}_y"]
    ], axis=1)

    dot = np.sum(vec_ab * vec_cb, axis=1)
    norm_ab = np.linalg.norm(vec_ab, axis=1)
    norm_cb = np.linalg.norm(vec_cb, axis=1)

    cos_angle = dot / (norm_ab * norm_cb + 1e-6)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)

all_dfs = []

for file in os.listdir(JSON_DIR):
    if file.endswith(".json"):
        filepath = os.path.join(JSON_DIR, file)
        origen = os.path.splitext(file)[0]
        df = json_to_dataframe(filepath, origen)
        all_dfs.append(df)

df_all = pd.concat(all_dfs, ignore_index=True)
print(f"Dataset cargado: {df_all.shape[0]} muestras, {df_all.shape[1]} columnas")

print("Extrayendo características geométricas...")

df_all["distancia_cadera_rodilla"] = distancia(df_all, "point_23", "point_25")
df_all["distancia_hombro_codo"] = distancia(df_all, "point_11", "point_13")

df_all["angulo_codo_izquierdo"] = angulo(df_all, "point_11", "point_13", "point_15")
df_all["angulo_rodilla_izquierda"] = angulo(df_all, "point_23", "point_25", "point_27")

df_all = df_all.sort_values(by=["origen", "frame"])

for col in ["distancia_cadera_rodilla", "distancia_hombro_codo",
            "angulo_codo_izquierdo", "angulo_rodilla_izquierda"]:
    delta_col = f"delta_{col}"
    df_all[delta_col] = df_all.groupby("origen")[col].diff().fillna(0)


df_all.to_csv(OUTPUT_CSV, index=False)
print(f"\nDataset completo guardado en: {OUTPUT_CSV}")
