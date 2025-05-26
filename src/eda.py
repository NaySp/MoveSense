import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Configuración y carga de datos ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data", "annotations", "keypoints_dataset.csv")

df = pd.read_csv(DATA_PATH)
print(f"Dataset cargado: {df.shape[0]} muestras, {df.shape[1]} columnas\n")

# --- Información general ---
print(df.info())
print(df.describe())
print("Conteo de clases original:")
print(df['label'].value_counts())

# --- Visualización inicial de clases ---
plt.figure(figsize=(6, 4))
sns.countplot(x='label', data=df)
plt.title("Distribución de clases original")
plt.show()

# --- Análisis de outliers inicial ---
num_cols = df.select_dtypes(include=[np.number]).columns
sample_cols = np.random.choice(num_cols, size=min(10, len(num_cols)), replace=False)
df[sample_cols].plot(kind='box', subplots=True, layout=(2, 5), figsize=(16, 6), sharex=False, sharey=False)
plt.suptitle("Boxplots de columnas numéricas (muestra aleatoria)")
plt.tight_layout()
plt.show()

# --- Correlación entre variables numéricas ---
corr = df[num_cols].corr()
plt.figure(figsize=(14, 10))
sns.heatmap(corr.iloc[:20, :20], annot=False, cmap='coolwarm')
plt.title("Mapa de calor de correlación (primeras 20 variables numéricas)")
plt.show()

# --- Eliminación de variables altamente correlacionadas (>0.98) ---
to_drop = set()
for i, j in zip(*np.where((corr.abs() > 0.98) & (corr.abs() < 1.0))):
    colname_i = num_cols[i]
    colname_j = num_cols[j]
    if colname_j not in to_drop:
        to_drop.add(colname_j)
df_reduced = df.drop(columns=list(to_drop))
print(f"Columnas eliminadas por alta correlación: {len(to_drop)}")
print(f"Shape después de eliminar correlación: {df_reduced.shape}")

# --- Eliminación de outliers usando IQR ---
for col in df_reduced.select_dtypes(include=[np.number]).columns:
    Q1 = df_reduced[col].quantile(0.25)
    Q3 = df_reduced[col].quantile(0.75)
    IQR = Q3 - Q1
    mask = (df_reduced[col] >= Q1 - 1.5 * IQR) & (df_reduced[col] <= Q3 + 1.5 * IQR)
    df_reduced = df_reduced[mask]
print(f"Shape después de eliminar outliers: {df_reduced.shape}")

# --- Conteo de clases después de limpieza ---
print("Conteo de clases después de eliminar outliers:")
print(df_reduced['label'].value_counts())

# --- Visualización de clases después de limpieza ---
plt.figure(figsize=(6, 4))
sns.countplot(x='label', data=df_reduced)
plt.title("Distribución de clases después de eliminar outliers")
plt.show()

# --- Boxplots después de limpieza ---
sample_cols = np.random.choice(df_reduced.select_dtypes(include=[np.number]).columns, size=min(10, len(df_reduced.select_dtypes(include=[np.number]).columns)), replace=False)
df_reduced[sample_cols].plot(kind='box', subplots=True, layout=(2, 5), figsize=(16, 6), sharex=False, sharey=False)
plt.suptitle("Boxplots después de eliminar outliers (muestra aleatoria)")
plt.tight_layout()
plt.show()


# --- Guarda dataset limpio y balanceado para entrenamiento ---
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "..", "data", "annotations", "keypoints_dataset_clean.csv")
df_reduced.to_csv(OUTPUT_PATH, index=False)
print(f"Dataset limpio guardado en: {OUTPUT_PATH}")