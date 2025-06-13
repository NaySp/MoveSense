import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data", "annotations", "keypoints_dataset.csv")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "..", "data", "annotations", "keypoints_dataset_clean.csv")

PROTECTED_COLS = {
    "origen", "label",
    "distancia_cadera_rodilla", "distancia_hombro_codo",
    "angulo_codo_izquierdo", "angulo_rodilla_izquierda"
}

def load_data(path):
    """Carga datos con validaciones detalladas"""
    try:
        if not os.path.exists(path):
            print(f"❌ Error: El archivo {path} no existe")
            return None
            
        df = pd.read_csv(path)
        print(f"✅ Dataset cargado exitosamente")
        print(f"📊 Dimensiones: {df.shape[0]} filas × {df.shape[1]} columnas")
        
        # Verificar columnas protegidas
        missing_protected = [col for col in PROTECTED_COLS if col not in df.columns]
        if missing_protected:
            print(f"⚠️  ADVERTENCIA: Columnas protegidas faltantes: {missing_protected}")
        else:
            print(f"✅ Todas las columnas protegidas están presentes")
            
        return df
    except Exception as e:
        print(f"❌ Error al cargar el dataset: {e}")
        return None

def validate_data_quality(df):
    """Validación completa de calidad de datos"""
    print("\n" + "="*60)
    print("🔍 VALIDACIÓN DE CALIDAD DE DATOS")
    print("="*60)
    
    # 1. Información básica
    print(f"\n📈 Información del dataset:")
    print(f"   • Filas: {df.shape[0]:,}")
    print(f"   • Columnas: {df.shape[1]:,}")
    print(f"   • Memoria: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # 2. Tipos de datos
    print(f"\n📊 Tipos de datos:")
    type_counts = df.dtypes.value_counts()
    for dtype, count in type_counts.items():
        print(f"   • {dtype}: {count} columnas")
    
    # 3. Valores faltantes
    missing_data = df.isnull().sum()
    missing_cols = missing_data[missing_data > 0]
    
    if len(missing_cols) > 0:
        print(f"\n⚠️  Valores faltantes encontrados:")
        for col, count in missing_cols.items():
            percentage = (count / len(df)) * 100
            print(f"   • {col}: {count} ({percentage:.2f}%)")
    else:
        print(f"\n✅ No hay valores faltantes")
    
    # 4. Duplicados
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"\n⚠️  Filas duplicadas: {duplicates}")
    else:
        print(f"\n✅ No hay filas duplicadas")
    
    # 5. Valores únicos por columna
    print(f"\n📊 Análisis de unicidad:")
    for col in df.columns:
        unique_count = df[col].nunique()
        total_count = len(df)
        uniqueness = (unique_count / total_count) * 100
        
        if uniqueness < 1:
            print(f"   ⚠️  {col}: {unique_count} valores únicos ({uniqueness:.2f}%) - Posible columna constante")
        elif uniqueness == 100:
            print(f"   ℹ️  {col}: Todos los valores son únicos")
    
    return missing_data, duplicates

def analyze_target_variable(df):
    """Análisis detallado de la variable objetivo"""
    print("\n" + "="*60)
    print("🎯 ANÁLISIS DE VARIABLE OBJETIVO")
    print("="*60)
    
    if 'label' not in df.columns:
        print("❌ No se encontró la columna 'label'")
        return None
    
    # Distribución de clases
    class_counts = df['label'].value_counts()
    print(f"\n📊 Distribución de clases:")
    for clase, count in class_counts.items():
        percentage = (count / len(df)) * 100
        print(f"   • {clase}: {count} muestras ({percentage:.2f}%)")
    
    # Detección de desbalance
    max_class = class_counts.max()
    min_class = class_counts.min()
    imbalance_ratio = max_class / min_class
    
    if imbalance_ratio > 3:
        print(f"\n⚠️  DESBALANCE DETECTADO:")
        print(f"   • Ratio de desbalance: {imbalance_ratio:.2f}:1")
        print(f"   • Clase mayoritaria: {class_counts.idxmax()} ({max_class} muestras)")
        print(f"   • Clase minoritaria: {class_counts.idxmin()} ({min_class} muestras)")
    else:
        print(f"\n✅ Distribución de clases relativamente balanceada")
    
    return class_counts

def analyze_numerical_features(df):
    """Análisis detallado de características numéricas"""
    print("\n" + "="*60)
    print("🔢 ANÁLISIS DE CARACTERÍSTICAS NUMÉRICAS")
    print("="*60)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(f"\n📊 Columnas numéricas encontradas: {len(numeric_cols)}")
    
    # Estadísticas descriptivas
    desc_stats = df[numeric_cols].describe()
    
    # Detectar posibles anomalías
    problematic_cols = []
    
    for col in numeric_cols:
        print(f"\n📈 {col}:")
        stats = desc_stats[col]
        
        # Verificar valores infinitos o NaN
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            print(f"   ⚠️  Valores infinitos: {inf_count}")
            problematic_cols.append(col)
        
        # Verificar rango de valores
        min_val, max_val = stats['min'], stats['max']
        print(f"   • Rango: [{min_val:.4f}, {max_val:.4f}]")
        print(f"   • Media: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        # Detectar valores constantes
        if stats['std'] == 0:
            print(f"   ⚠️  Columna constante (std = 0)")
            problematic_cols.append(col)
        
        # Detectar outliers extremos
        q1, q3 = stats['25%'], stats['75%']
        iqr = q3 - q1
        lower_bound = q1 - 3 * iqr  # 3 IQR en lugar de 1.5 para outliers extremos
        upper_bound = q3 + 3 * iqr
        
        extreme_outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        if extreme_outliers > 0:
            percentage = (extreme_outliers / len(df)) * 100
            print(f"   ⚠️  Outliers extremos: {extreme_outliers} ({percentage:.2f}%)")
    
    if problematic_cols:
        print(f"\n⚠️  Columnas problemáticas detectadas: {len(problematic_cols)}")
        for col in problematic_cols:
            print(f"   • {col}")
    
    return numeric_cols, problematic_cols

def analyze_small_dataset_strategy(df):
    """Análisis específico para datasets pequeños"""
    print("\n" + "="*60)
    print("📊 ESTRATEGIA PARA DATASET PEQUEÑO")
    print("="*60)
    
    total_samples = len(df)
    print(f"\n📊 Tamaño del dataset: {total_samples} muestras")
    
    # Evaluación del tamaño
    if total_samples < 1000:
        print(f"⚠️  Dataset PEQUEÑO detectado (< 1000 muestras)")
        print(f"📋 Recomendaciones:")
        print(f"   • Mantener TODAS las características disponibles")
        print(f"   • Usar validación cruzada en lugar de split fijo")
        print(f"   • Considerar técnicas de aumento de datos")
        print(f"   • Aplicar regularización en modelos")
        print(f"   • Usar ensemble methods para mejor generalización")
    elif total_samples < 5000:
        print(f"⚡ Dataset MEDIANO detectado (1K-5K muestras)")
        print(f"📋 Recomendaciones:")
        print(f"   • Ser conservador con eliminación de características")
        print(f"   • Validación cruzada k-fold (k=5 o k=10)")
        print(f"   • Monitorear overfitting cuidadosamente")
    else:
        print(f"✅ Dataset GRANDE detectado (> 5K muestras)")
        print(f"📋 Estrategia estándar aplicable")
    
    # Análisis por clase
    if 'label' in df.columns:
        class_counts = df['label'].value_counts()
        min_class_size = class_counts.min()
        
        print(f"\n📊 Análisis por clase:")
        print(f"   • Clase más pequeña: {min_class_size} muestras")
        
        if min_class_size < 50:
            print(f"   ⚠️  CLASE MINORITARIA CRÍTICA (< 50 muestras)")
            print(f"   📋 Considerar:")
            print(f"      • Técnicas de balanceo (SMOTE, undersampling)")
            print(f"      • Stratified sampling obligatorio")
            print(f"      • Métricas balanceadas (F1, AUC)")
        elif min_class_size < 100:
            print(f"   ⚡ Clase minoritaria pequeña (< 100 muestras)")
            print(f"   📋 Usar stratified split y validación cruzada")
        
        # Ratio de desbalance
        max_class_size = class_counts.max()
        imbalance_ratio = max_class_size / min_class_size
        print(f"   • Ratio de desbalance: {imbalance_ratio:.2f}:1")
    
    return total_samples < 1000

def plot_enhanced_visualizations(df):
    """Visualizaciones mejoradas con más información"""
    print("\n" + "="*60)
    print("📊 GENERANDO VISUALIZACIONES")
    print("="*60)
    
    # 1. Distribución de clases mejorada
    if 'label' in df.columns:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Gráfico de barras
        class_counts = df['label'].value_counts()
        ax1.bar(class_counts.index, class_counts.values, color='skyblue', alpha=0.8)
        ax1.set_title('Distribución de Clases')
        ax1.set_xlabel('Clase')
        ax1.set_ylabel('Cantidad')
        
        # Agregar etiquetas de porcentaje
        total = len(df)
        for i, (clase, count) in enumerate(class_counts.items()):
            percentage = (count / total) * 100
            ax1.text(i, count + total*0.01, f'{percentage:.1f}%', 
                    ha='center', va='bottom', fontweight='bold')
        
        # Gráfico de torta
        ax2.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%', 
                startangle=90, colors=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
        ax2.set_title('Proporción de Clases')
        
        plt.tight_layout()
        plt.show()
    
    # 2. Boxplots con información de outliers
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        n_cols = min(10, len(numeric_cols))
        sample_cols = np.random.choice(numeric_cols, size=n_cols, replace=False)
        
        fig, axes = plt.subplots(2, 5, figsize=(18, 8))
        axes = axes.ravel()
        
        for i, col in enumerate(sample_cols):
            ax = axes[i]
            box_plot = ax.boxplot(df[col].dropna(), patch_artist=True)
            box_plot['boxes'][0].set_facecolor('lightblue')
            ax.set_title(f'{col}\n(n={df[col].notna().sum()})')
            ax.tick_params(axis='x', labelbottom=False)
            
            # Agregar estadísticas
            q1, median, q3 = df[col].quantile([0.25, 0.5, 0.75])
            ax.text(0.02, 0.98, f'Q1: {q1:.2f}\nMed: {median:.2f}\nQ3: {q3:.2f}', 
                   transform=ax.transAxes, verticalalignment='top', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.suptitle('Distribución de Variables Numéricas (Muestra)')
        plt.tight_layout()
        plt.show()

def correlation_analysis(df, threshold=0.98):
    """Análisis de correlación mejorado"""
    print("\n" + "="*60)
    print("🔗 ANÁLISIS DE CORRELACIÓN")
    print("="*60)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        print("❌ No hay suficientes columnas numéricas para análisis de correlación")
        return None, []
    
    corr_matrix = df[numeric_cols].corr()
    
    # Encontrar correlaciones altas
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_val = abs(corr_matrix.iloc[i, j])
            if corr_val > threshold:
                col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                high_corr_pairs.append((col1, col2, corr_val))
    
    if high_corr_pairs:
        print(f"\n⚠️  Correlaciones altas encontradas (|r| > {threshold}):")
        for col1, col2, corr_val in high_corr_pairs:
            print(f"   • {col1} ↔ {col2}: {corr_val:.4f}")
    else:
        print(f"\n✅ No se encontraron correlaciones extremas (|r| > {threshold})")
    
    # Visualización de correlación
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', 
                center=0, square=True, linewidths=0.5)
    plt.title('Matriz de Correlación (Triángulo Superior)')
    plt.tight_layout()
    plt.show()
    
    return corr_matrix, high_corr_pairs

def clean_data_with_validation(df, corr_matrix, threshold=0.98, remove_correlated=False):
    """Limpieza de datos con validación en cada paso"""
    print("\n" + "="*60)
    print("🧹 PROCESO DE LIMPIEZA")
    print("="*60)
    
    initial_shape = df.shape
    print(f"\n📊 Datos iniciales: {initial_shape[0]} filas × {initial_shape[1]} columnas")
    
    # 1. Análisis de correlación (sin eliminar por defecto)
    to_drop = set()
    if corr_matrix is not None and remove_correlated:
        cols = corr_matrix.columns
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    col_j = cols[j]
                    if col_j not in PROTECTED_COLS:
                        to_drop.add(col_j)
    
    if to_drop and remove_correlated:
        print(f"\n🔄 Eliminando {len(to_drop)} columnas por alta correlación:")
        for col in sorted(to_drop):
            print(f"   • {col}")
        df_reduced = df.drop(columns=list(to_drop))
        print(f"📊 Después de eliminar correlación: {df_reduced.shape[0]} filas × {df_reduced.shape[1]} columnas")
    else:
        print(f"\n✅ MANTENIENDO todas las columnas (recomendado para datasets pequeños)")
        if corr_matrix is not None:
            high_corr_count = 0
            cols = corr_matrix.columns
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    if abs(corr_matrix.iloc[i, j]) > threshold:
                        high_corr_count += 1
            if high_corr_count > 0:
                print(f"   • Se detectaron {high_corr_count} pares con alta correlación")
                print(f"   • Manteniendo para preservar información con pocos datos")
        df_reduced = df.copy()
        print(f"📊 Columnas preservadas: {df_reduced.shape[0]} filas × {df_reduced.shape[1]} columnas")
    
    # 2. Eliminar outliers con conteo detallado (método más conservador)
    print(f"\n🔄 Eliminando outliers (método conservador)...")
    before_outliers = df_reduced.shape[0]
    
    outlier_counts = {}
    for col in df_reduced.select_dtypes(include=[np.number]).columns:
        before_col = df_reduced.shape[0]
        Q1 = df_reduced[col].quantile(0.25)
        Q3 = df_reduced[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Usar límites MÁS conservadores para preservar datos
        # 2.0 * IQR en lugar de 1.5 para ser menos agresivo
        lower_bound = Q1 - 2.0 * IQR
        upper_bound = Q3 + 2.0 * IQR
        
        # Contar outliers antes de eliminar
        outliers_count = ((df_reduced[col] < lower_bound) | (df_reduced[col] > upper_bound)).sum()
        outlier_counts[col] = outliers_count
        
        # Solo eliminar outliers extremos
        mask = (df_reduced[col] >= lower_bound) & (df_reduced[col] <= upper_bound)
        df_reduced = df_reduced[mask]
        
        removed = before_col - df_reduced.shape[0]
        if removed > 0:
            print(f"   • {col}: {removed} outliers eliminados (de {outliers_count} detectados)")
    
    after_outliers = df_reduced.shape[0]
    total_outliers_removed = before_outliers - after_outliers
    
    # Reporte conservador
    print(f"\n📊 Estrategia conservadora aplicada:")
    print(f"   • Límites: Q1 - 2.0*IQR, Q3 + 2.0*IQR (menos agresivo)")
    print(f"   • Outliers eliminados: {total_outliers_removed} filas ({(total_outliers_removed/before_outliers)*100:.2f}%)")
    print(f"   • Datos preservados: {after_outliers} filas ({(after_outliers/before_outliers)*100:.2f}%)")
    print(f"📊 Datos finales: {df_reduced.shape[0]} filas × {df_reduced.shape[1]} columnas")
    
    # 3. Verificar integridad después de limpieza
    print(f"\n🔍 Verificación post-limpieza:")
    
    # Verificar que las columnas protegidas siguen presentes
    missing_protected = [col for col in PROTECTED_COLS if col not in df_reduced.columns]
    if missing_protected:
        print(f"❌ ADVERTENCIA: Columnas protegidas perdidas: {missing_protected}")
    else:
        print(f"✅ Todas las columnas protegidas preservadas")
    
    # Verificar distribución de clases
    if 'label' in df_reduced.columns:
        original_classes = df['label'].value_counts()
        final_classes = df_reduced['label'].value_counts()
        
        print(f"\n📊 Cambios en distribución de clases:")
        for clase in original_classes.index:
            original_count = original_classes[clase]
            final_count = final_classes.get(clase, 0)
            reduction = ((original_count - final_count) / original_count) * 100
            print(f"   • {clase}: {original_count} → {final_count} (-{reduction:.1f}%)")
    
    return df_reduced

def save_with_validation(df, output_path):
    """Guardar datos con validación final"""
    print("\n" + "="*60)
    print("💾 GUARDANDO DATOS LIMPIOS")
    print("="*60)
    
    # Verificación final
    required_cols = list(PROTECTED_COLS)
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"❌ ERROR: Faltan columnas esenciales: {missing_cols}")
        print("❌ No se guardará el archivo")
        return False
    
    try:
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        df.to_csv(output_path, index=False)
        print(f"✅ Dataset limpio guardado exitosamente")
        print(f"📁 Ubicación: {output_path}")
        print(f"📊 Dimensiones finales: {df.shape[0]} filas × {df.shape[1]} columnas")
        
        # Verificar que el archivo se guardó correctamente
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / 1024**2  # MB
            print(f"📦 Tamaño del archivo: {file_size:.2f} MB")
            
            # Verificar que se puede leer
            test_df = pd.read_csv(output_path)
            if test_df.shape == df.shape:
                print(f"✅ Verificación de integridad: EXITOSA")
            else:
                print(f"⚠️  Advertencia: Dimensiones no coinciden al releer")
        
        return True
        
    except Exception as e:
        print(f"❌ Error al guardar: {e}")
        return False

def main():
    """Función principal con flujo de validación completo"""
    print("="*80)
    print("🚀 INICIANDO ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
    print("="*80)
    
    # 1. Cargar datos
    df = load_data(DATA_PATH)
    if df is None:
        return
    
    # 2. Validación inicial
    missing_data, duplicates = validate_data_quality(df)
    
    # 3. Análisis de variable objetivo
    class_distribution = analyze_target_variable(df)
    
    # 4. Análisis de características numéricas
    numeric_cols, problematic_cols = analyze_numerical_features(df)
    
    # 5. Estrategia para dataset pequeño - ESTA LÍNEA ESTABA MAL UBICADA
    is_small_dataset = analyze_small_dataset_strategy(df)
    
    # 6. Visualizaciones
    plot_enhanced_visualizations(df)
    
    # 7. Análisis de correlación
    corr_matrix, high_corr_pairs = correlation_analysis(df)
    
    # 8. Limpieza de datos (conservadora para datasets pequeños)
    df_clean = clean_data_with_validation(df, corr_matrix, remove_correlated=False)
    
    # 9. Análisis post-limpieza
    print("\n" + "="*60)
    print("📊 ANÁLISIS POST-LIMPIEZA")
    print("="*60)
    
    if 'label' in df_clean.columns:
        final_distribution = df_clean['label'].value_counts()
        print(f"\n📊 Distribución final de clases:")
        for clase, count in final_distribution.items():
            percentage = (count / len(df_clean)) * 100
            print(f"   • {clase}: {count} muestras ({percentage:.2f}%)")
    
    # 10. Guardar datos limpios
    success = save_with_validation(df_clean, OUTPUT_PATH)
    
    if success:
        print("\n" + "="*80)
        print("✅ EDA COMPLETADO EXITOSAMENTE")
        print("="*80)
        print(f"📊 Resumen final:")
        print(f"   • Datos originales: {df.shape[0]} filas × {df.shape[1]} columnas")
        print(f"   • Datos limpios: {df_clean.shape[0]} filas × {df_clean.shape[1]} columnas")
        print(f"   • Reducción: {((df.shape[0] - df_clean.shape[0]) / df.shape[0]) * 100:.2f}% filas")
        print(f"   • Archivo guardado: {OUTPUT_PATH}")
    else:
        print("\n❌ EDA finalizado con errores")

if __name__ == "__main__":
    main()