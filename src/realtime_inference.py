import cv2
import mediapipe as mp
import numpy as np
from joblib import load
import os
from collections import deque
from math import acos, degrees
from numpy.linalg import norm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "demo", "gradientboost_pose_model.pkl")
FEATURES_PATH = os.path.join(SCRIPT_DIR, "..", "demo", "feature_names.txt")

model = load(MODEL_PATH)
with open(FEATURES_PATH) as f:
    feature_names = [line.strip() for line in f]

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

delta_hist = deque(maxlen=2)


def calc_angle(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cos_angle = np.dot(ba, bc) / (norm(ba) * norm(bc) + 1e-6)
    return degrees(acos(np.clip(cos_angle, -1.0, 1.0)))


def calc_distance(a, b):
    return norm(np.array(a) - np.array(b))


def extract_all_features(landmarks):
    features = {}

    for i, lm in enumerate(landmarks):
        for axis in ['x', 'y', 'z']:
            key = f"point_{i}_{axis}"
            features[key] = getattr(lm, axis, 0.0)

    features["distancia_cadera_rodilla"] = calc_distance(
        [landmarks[23].x, landmarks[23].y],
        [landmarks[25].x, landmarks[25].y],
    )
    features["angulo_codo_izquierdo"] = calc_angle(
        [landmarks[11].x, landmarks[11].y],
        [landmarks[13].x, landmarks[13].y],
        [landmarks[15].x, landmarks[15].y],
    )
    features["angulo_rodilla_izquierda"] = calc_angle(
        [landmarks[23].x, landmarks[23].y],
        [landmarks[25].x, landmarks[25].y],
        [landmarks[27].x, landmarks[27].y],
    )

    return features


def main():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        if result.pose_landmarks:
            mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            current_feats = extract_all_features(result.pose_landmarks.landmark)

            delta_feats = {}
            if delta_hist:
                prev_feats = delta_hist[-1]
                for key in current_feats:
                    delta_feats[f"delta_{key}"] = current_feats[key] - prev_feats.get(key, 0)
            else:
                for key in current_feats:
                    delta_feats[f"delta_{key}"] = 0.0

            delta_hist.append(current_feats)

            all_feats = {**current_feats, **delta_feats}

            input_vector = [all_feats.get(f, 0.0) for f in feature_names]
            ordered_input = np.array(input_vector).reshape(1, -1)

            try:
                pred = model.predict(ordered_input)[0]
                prob = model.predict_proba(ordered_input).max()

                label = f"{pred} ({prob:.2f})"
                cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2, cv2.LINE_AA)
            except Exception as e:
                print("Error en predicción:", e)

        cv2.imshow("Predicción en vivo", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()