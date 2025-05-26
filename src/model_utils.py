import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data", "annotations", "keypoints_dataset.csv")

df = pd.read_csv(DATA_PATH)
print(f"Dataset cargado: {df.shape[0]} muestras, {df.shape[1]} columnas\n")

cols_vis = [col for col in df.columns if "_visibility" in col]
df.drop(columns=cols_vis, inplace=True)

varianzas = df.var(numeric_only=True)
cols_baja_varianza = varianzas[varianzas < 0.0001].index
df.drop(columns=cols_baja_varianza, inplace=True)

X = df.drop(columns=["label", "frame", "origen"], errors="ignore")
y = df["label"]
groups = df["origen"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

cv = GroupKFold(n_splits=3)

models = {
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [100],
            "max_depth": [10, None]
        }
    },
    "SVC": {
        "model": SVC(),
        "params": {
            "C": [1, 10],
            "kernel": ["linear", "rbf"]
        }
    },
    "KNN": {
        "model": KNeighborsClassifier(),
        "params": {
            "n_neighbors": [3, 5, 7]
        }
    },
    "Gradient Boosting": {
        "model": GradientBoostingClassifier(random_state=42),
        "params": {
            "n_estimators": [100],
            "learning_rate": [0.1, 0.05],
            "max_depth": [3, 5]
        }
    },
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=1000),
        "params": {
            "C": [0.1, 1, 10]
        }
    }
}

for name, m in models.items():
    print(f"\n🔍 {name}...")
    grid = GridSearchCV(
        m["model"], m["params"],
        cv=cv.split(X_scaled, y, groups),
        scoring="f1_macro",
        n_jobs=-1,
        verbose=0
    )
    grid.fit(X_scaled, y)
    print(f"Mejor modelo: {grid.best_estimator_}")
    y_pred = grid.predict(X_scaled)
    print(classification_report(y, y_pred))
