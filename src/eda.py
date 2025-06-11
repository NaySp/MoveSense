import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data", "annotations", "keypoints_dataset.csv")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "..", "data", "annotations", "keypoints_dataset_clean.csv")

PROTECTED_COLS = {
    "origen", "label",
    "distancia_cadera_rodilla", "distancia_hombro_codo",
    "angulo_codo_izquierdo", "angulo_rodilla_izquierda"
}

def load_data(path):
    try:
        df = pd.read_csv(path)
        print(f"Dataset cargado: {df.shape[0]} muestras, {df.shape[1]} columnas\n")
        return df
    except Exception as e:
        print(f"Error al cargar el dataset: {e}")
        return None

def describe_data(df):
    print(df.info())
    print(df.describe())
    print("\n📊 Conteo de clases original:")
    print(df['label'].value_counts())

def plot_class_distribution(df, title="Distribución de clases"):
    plt.figure(figsize=(6, 4))
    sns.countplot(x='label', data=df)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_boxplots(df, n=10):
    num_cols = df.select_dtypes(include=[np.number]).columns
    sample_cols = np.random.choice(num_cols, size=min(n, len(num_cols)), replace=False)
    df[sample_cols].plot(kind='box', subplots=True, layout=(2, 5), figsize=(16, 6), sharex=False, sharey=False)
    plt.suptitle("Boxplots de columnas numéricas (muestra aleatoria)")
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(df, top_n=20):
    num_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[num_cols].corr()
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr.iloc[:top_n, :top_n], cmap='coolwarm', annot=False)
    plt.title("Mapa de calor de correlación (primeras 20 variables numéricas)")
    plt.tight_layout()
    plt.show()
    return corr, num_cols

def drop_highly_correlated(df, corr_matrix, threshold=0.98):
    to_drop = set()
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                col_j = cols[j]
                if col_j not in PROTECTED_COLS:
                    to_drop.add(col_j)
    print(f"\n🧹 Columnas eliminadas por alta correlación: {len(to_drop)}")
    return df.drop(columns=list(to_drop)), to_drop

def remove_outliers(df):
    before = df.shape[0]
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        mask = (df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)
        df = df[mask]
    after = df.shape[0]
    print(f"\nOutliers eliminados: {before - after} filas")
    return df

def check_and_save(df, output_path):
    required = list(PROTECTED_COLS)
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"Faltan columnas esenciales para entrenamiento: {missing}")
    else:
        df.to_csv(output_path, index=False)
        print(f"\nDataset limpio guardado en: {output_path}")

def main():
    df = load_data(DATA_PATH)
    if df is None:
        return

    describe_data(df)
    plot_class_distribution(df)

    plot_boxplots(df)
    corr_matrix, num_cols = plot_correlation_heatmap(df)

    df_reduced, dropped_cols = drop_highly_correlated(df, corr_matrix)
    print(f"Shape después de eliminar correlación: {df_reduced.shape}")

    df_clean = remove_outliers(df_reduced)
    print(f"Shape final después de eliminar outliers: {df_clean.shape}")

    print("\nConteo de clases después de limpieza:")
    print(df_clean['label'].value_counts())

    plot_class_distribution(df_clean, title="Distribución de clases final")

    check_and_save(df_clean, OUTPUT_PATH)

if __name__ == "__main__":
    main()