import os
import cv2
import json
import mediapipe as mp
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.join(SCRIPT_DIR, "..", "data", "raw_30fps")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "data", "processed")

os.makedirs(OUTPUT_DIR, exist_ok=True)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)

interval_seconds = 1.0

video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith((".mp4", ".MOV"))]
print(f"Procesando {len(video_files)} videos...")

for filename in tqdm(video_files):
    filepath = os.path.join(VIDEO_DIR, filename)
    label = filename.split("_")[0].lower()

    cap = cv2.VideoCapture(filepath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    frame_interval = int(fps * interval_seconds)

    print(f"\n{filename} | Duración: {duration:.2f}s | FPS: {fps:.2f} | Frames: {frame_count} | Intervalo: {frame_interval} frames")

    keypoints_data = []
    frame_idx = 0

    while cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
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
                "time_seconds": round(frame_idx / fps, 2),
                "label": label,
                "keypoints": keypoints
            })

        frame_idx += frame_interval

    cap.release()

    output_file = os.path.join(OUTPUT_DIR, f"keypoints_{os.path.splitext(filename)[0]}.json")
    with open(output_file, "w") as f:
        json.dump(keypoints_data, f, indent=2)

print("\nExtracción finalizada. Archivos guardados en:", OUTPUT_DIR)