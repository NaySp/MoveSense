import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from joblib import load
from collections import deque

# Cargar modelo y scaler
model = load("demo/gradientboost_pose_model.pkl")
scaler = load("demo/scaler.pkl")  # Solo si fue usado al entrenar

# Leer nombres de features
with open("demo/feature_names.txt") as f:
    expected_features = [line.strip() for line in f]

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

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

    feats = {
        "point_7_z": punto(7).z,
        "point_8_z": punto(8).z,
        "point_11_z": punto(11).z,
        "point_13_z": punto(13).z,
        "point_23_x": punto(23).x,
        "point_24_x": punto(24).x,
        "point_25_x": punto(25).x,
        "point_25_y": punto(25).y,
        "point_25_z": punto(25).z,
        "point_26_y": punto(26).y,
        "point_26_z": punto(26).z,
        "point_31_x": punto(31).x,
        "point_31_y": punto(31).y,
        "point_31_z": punto(31).z,
        "point_32_x": punto(32).x,
        "point_32_y": punto(32).y,
        "point_32_z": punto(32).z,
        "distancia_cadera_rodilla": distancia(punto(23), punto(25)),
        "angulo_codo_izquierdo": angulo(punto(11), punto(13), punto(15)),
        "angulo_rodilla_izquierda": angulo(punto(23), punto(25), punto(27))
    }
    return feats

# Activar cámara
cap = cv2.VideoCapture(0)
print("Presiona 'q' para salir.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        feats = extract_features(results.pose_landmarks.landmark)
        history.append(feats)

        if len(history) == 2:
            # Calcular deltas
            delta_feats = {
                f"delta_{k}": history[-1][k] - history[-2][k]
                for k in ["distancia_cadera_rodilla", "angulo_codo_izquierdo", "angulo_rodilla_izquierda"]
            }

            # Combinar features actuales + delta
            combined = {f: 0.0 for f in expected_features}
            combined.update({k: feats.get(k, 0.0) for k in combined})
            combined.update({k: delta_feats.get(k, 0.0) for k in combined})

            X = pd.DataFrame([combined])[expected_features]
            X_scaled = scaler.transform(X)
            pred = model.predict(X_scaled)[0]

            cv2.putText(frame, f"Acción: {pred}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.imshow("Predicción en tiempo real", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
