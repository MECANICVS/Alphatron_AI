"""With this script, Piper reads the output of Yolo"""
# To install Piper on Linux "pip install piper-tts", 
# On Windows, simply unzip piper_windows_amd64.zip from https://github.com/rhasspy/piper/releases/tag/2023.11.14-2
import time
import subprocess
import threading
import sounddevice as sd
import numpy as np
from ultralytics import YOLO

# Piper command to generate raw audio
piper_cmd = [
    "piper",
    "--model", "en_US-hfc_male-medium.onnx",
    "--output-raw"
]
# Replace en_US-hfc_male-medium.onnx with the voice of your choice

# Sample rate and format (ensure this matches Piper's output)
samplerate = 22050
dtype = np.int16  # Assuming Piper outputs 16-bit PCM

def stream_audio(text):
    with subprocess.Popen(piper_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1024) as p:
        # Send the text to Piper's stdin and close it to indicate end of input
        p.stdin.write(text.encode('utf-8'))
        p.stdin.close()
        # No need to add device, even on Rpi
        with sd.RawOutputStream(samplerate=samplerate, dtype=dtype, channels=1) as stream:
        #    print("Streaming Piper output...")
            while True:
                data = p.stdout.read(1024)  # Read Piper's raw audio output
                if not data:
                    break
                # Convert the raw byte data to a numpy array
                audio_data = np.frombuffer(data, dtype=dtype)
                stream.write(audio_data)  # Play the audio in real-time


# Function to run YOLO and detect objects
def run_yolo():
    model = YOLO("vision.pt")
    class_names = model.names  # Get class names

    # Run inference on the source (camera feed or image)
    results = model.predict(source=0, imgsz=320, stream=True, verbose=False)

    for result in results:
        detected_objects = []
        for box in result.boxes:
            class_id = int(box.cls)
            detected_objects.append(class_names[class_id])  # Append detected class name

        if detected_objects:
            detected_text = ", ".join(detected_objects)
            print(detected_text)
            # Pass the detected text to Piper for speech output
            threading.Thread(target=stream_audio, args=(detected_text,), daemon=True).start()

        time.sleep(3)  # Add a small delay between detections

# Run YOLO and Piper concurrently using threading
if __name__ == "__main__":
    run_yolo()

