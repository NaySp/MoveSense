import os
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(SCRIPT_DIR, "..", "data", "raw")
output_dir = os.path.join(SCRIPT_DIR, "..", "data", "raw_30fps")
os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(input_dir):
    if file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        input_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_dir, file)
        print(f"Procesando {file}...")
        subprocess.run([
            "ffmpeg", "-y", "-i", input_path,
            "-r", "30",  # fuerza 30 fps
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "copy",
            output_path
        ], check=True)
print("Conversión completada.")