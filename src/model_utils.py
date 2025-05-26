import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# --- Carga de datos ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data", "annotations", "keypoints_dataset_clean.csv")

df = pd.read_csv(DATA_PATH)
print(f"Dataset cargado: {df.shape[0]} muestras, {df.shape[1]} columnas\n")

# --- Limpieza de columnas innecesarias ---
cols_vis = [col for col in df.columns if "_visibility" in col]
df.drop(columns=cols_vis, inplace=True)

varianzas = df.var(numeric_only=True)
cols_baja_varianza = varianzas[varianzas < 0.0001].index
df.drop(columns=cols_baja_varianza, inplace=True)

# --- División estratificada ---
X = df.drop(columns=["label", "frame", "origen"], errors="ignore")
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("Train class counts:\n", y_train.value_counts())
print("Test class counts:\n", y_test.value_counts())

# --- Definición de modelos y parámetros ---
models = {
    "Random Forest": {
        "model": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(random_state=42))
        ]),
        "params": {
            "classifier__n_estimators": [100],
            "classifier__max_depth": [10, None]
        }
    },
    "SVC": {
        "model": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC())
        ]),
        "params": {
            "classifier__C": [1, 10],
            "classifier__kernel": ["linear", "rbf"]
        }
    },
    "KNN": {
        "model": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier())
        ]),
        "params": {
            "classifier__n_neighbors": [3, 5, 7]
        }
    },
    "Gradient Boosting": {
        "model": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", GradientBoostingClassifier(random_state=42))
        ]),
        "params": {
            "classifier__n_estimators": [100],
            "classifier__learning_rate": [0.1, 0.05],
            "classifier__max_depth": [3, 5]
        }
    },
    "Logistic Regression": {
        "model": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000))
        ]),
        "params": {
            "classifier__C": [0.1, 1, 10]
        }
    }
}

# --- Validación cruzada estratificada y evaluación ---
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
best_models = {}

for name, m in models.items():
    print(f"\n{name}...")
    grid = GridSearchCV(
        m["model"], m["params"],
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=0
    )
    grid.fit(X_train, y_train)
    best_models[name] = grid.best_estimator_
    print(f"Mejor modelo: {grid.best_estimator_}")

    # Evaluación en entrenamiento
    y_train_pred = grid.predict(X_train)
    print("Resultados en datos de entrenamiento:")
    print(classification_report(y_train, y_train_pred))

    # Evaluación en prueba
    y_test_pred = grid.predict(X_test)
    print("Resultados en datos de prueba:")
    print(classification_report(y_test, y_test_pred))
