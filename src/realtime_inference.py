import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from joblib import load
from collections import deque

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "demo", "gradientboost_pose_model.pkl")
SCALER_PATH = os.path.join(SCRIPT_DIR, "..", "demo", "scaler.pkl")
FEATURES_PATH = os.path.join(SCRIPT_DIR, "..", "demo", "feature_names.txt")

model = load(MODEL_PATH)
scaler = load(SCALER_PATH)
with open(FEATURES_PATH) as f:
    expected_features = [line.strip() for line in f]

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)

history = deque(maxlen=2)

def extract_features(landmarks):
    def punto(idx):
        return landmarks[idx]

    def distancia(p1, p2):
        return np.linalg.norm(np.array([p1.x - p2.x, p1.y - p2.y]))

    def angulo(a, b, c):
        ba = np.array([a.x - b.x, a.y - b.y])
        bc = np.array([c.x - b.x, c.y - b.y])
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    return {
        "distancia_cadera_rodilla": distancia(punto(23), punto(25)),
        "distancia_hombro_codo": distancia(punto(11), punto(13)),
        "angulo_codo_izquierdo": angulo(punto(11), punto(13), punto(15)),
        "angulo_rodilla_izquierda": angulo(punto(23), punto(25), punto(27)),
    }

cap = cv2.VideoCapture(0)
print("Cámara activada. Presiona 'q' para salir.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        feats = extract_features(results.pose_landmarks.landmark)
        history.append(feats)

        if len(history) == 2:
            delta_feats = {
                f"delta_{k}": history[-1][k] - history[-2][k]
                for k in feats
            }

            combined = {feat: 0.0 for feat in expected_features}
            combined.update({k: v for k, v in feats.items() if k in combined})
            combined.update({k: v for k, v in delta_feats.items() if k in combined})

            X_input = pd.DataFrame([combined])[expected_features]
            X_scaled = scaler.transform(X_input)
            pred = model.predict(X_scaled)[0]

            cv2.putText(frame, f"Accion: {pred}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.imshow("Prediccion en tiempo real", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()