import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data", "annotations", "keypoints_dataset.csv")

df = pd.read_csv(DATA_PATH)
print(f"Dataset cargado: {df.shape[0]} muestras, {df.shape[1]} columnas\n")

print(df.info())
print(df.describe())
print("\nConteo de clases original:")
print(df['label'].value_counts())

plt.figure(figsize=(6, 4))
sns.countplot(x='label', data=df)
plt.title("Distribución de clases original")
plt.tight_layout()
plt.show()

num_cols = df.select_dtypes(include=[np.number]).columns
sample_cols = np.random.choice(num_cols, size=min(10, len(num_cols)), replace=False)
df[sample_cols].plot(kind='box', subplots=True, layout=(2, 5), figsize=(16, 6), sharex=False, sharey=False)
plt.suptitle("Boxplots de columnas numéricas (muestra aleatoria)")
plt.tight_layout()
plt.show()

corr = df[num_cols].corr()
plt.figure(figsize=(14, 10))
sns.heatmap(corr.iloc[:20, :20], cmap='coolwarm')
plt.title("Mapa de calor de correlación (primeras 20 variables numéricas)")
plt.tight_layout()
plt.show()

protected_cols = set([
    "origen", "label",
    "distancia_cadera_rodilla", "distancia_hombro_codo",
    "angulo_codo_izquierdo", "angulo_rodilla_izquierda"
])
to_drop = set()
for i, j in zip(*np.where((corr.abs() > 0.98) & (corr.abs() < 1.0))):
    col_i, col_j = num_cols[i], num_cols[j]
    if col_j not in protected_cols:
        to_drop.add(col_j)

df_reduced = df.drop(columns=list(to_drop))
print(f"\nColumnas eliminadas por alta correlación: {len(to_drop)}")
print(f"Shape después de eliminar correlación: {df_reduced.shape}")

for col in df_reduced.select_dtypes(include=[np.number]).columns:
    Q1 = df_reduced[col].quantile(0.25)
    Q3 = df_reduced[col].quantile(0.75)
    IQR = Q3 - Q1
    mask = (df_reduced[col] >= Q1 - 1.5 * IQR) & (df_reduced[col] <= Q3 + 1.5 * IQR)
    df_reduced = df_reduced[mask]

print(f"\nShape después de eliminar outliers: {df_reduced.shape}")

print("\nConteo de clases después de limpieza:")
print(df_reduced['label'].value_counts())

plt.figure(figsize=(6, 4))
sns.countplot(x='label', data=df_reduced)
plt.title("Distribución de clases final")
plt.tight_layout()
plt.show()

required = [
    "origen", "label",
    "distancia_cadera_rodilla", "distancia_hombro_codo",
    "angulo_codo_izquierdo", "angulo_rodilla_izquierda"
]
missing = [col for col in required if col not in df_reduced.columns]
if missing:
    print(f"Faltan columnas esenciales para entrenamiento: {missing}")
else:
    OUTPUT_PATH = os.path.join(SCRIPT_DIR, "..", "data", "annotations", "keypoints_dataset_clean.csv")
    df_reduced.to_csv(OUTPUT_PATH, index=False)
    print(f"\nDataset limpio guardado en: {OUTPUT_PATH}")