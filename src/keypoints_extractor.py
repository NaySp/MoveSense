import os
import cv2
import json
import mediapipe as mp
from tqdm import tqdm

VIDEO_DIR = "../data/raw/"  
OUTPUT_DIR = "../data/processed/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)

video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")]

print(f"Procesando {len(video_files)} videos...")

for filename in tqdm(video_files):
    filepath = os.path.join(VIDEO_DIR, filename)
    label = filename.split("_")[0].lower()

    cap = cv2.VideoCapture(filepath)
    frame_idx = 0
    keypoints_data = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            keypoints = {
                f"point_{i}": {
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z,
                    "visibility": lm.visibility
                } for i, lm in enumerate(results.pose_landmarks.landmark)
            }

            keypoints_data.append({
                "frame": frame_idx,
                "label": label,
                "keypoints": keypoints
            })

        frame_idx += 1

    cap.release()

    output_file = os.path.join(OUTPUT_DIR, f"keypoints_{os.path.splitext(filename)[0]}.json")
    with open(output_file, "w") as f:
        json.dump(keypoints_data, f, indent=2)

print("✅ Extracción finalizada. Archivos guardados en:", OUTPUT_DIR)
