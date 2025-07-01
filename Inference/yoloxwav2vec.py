"""If the word you pronounce matches what Yolo sees, you get a vocal confirmation"""
import torch
import numpy as np
import soundfile as sf
import sounddevice as sd
import time
import subprocess
import threading
from ultralytics import YOLO
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# Parameters
freq = 16000
channels = 1
energy_threshold = 0.02 # recommend between 0.01 and 0.05
filename = "Yes.wav" #just an example audio file for confirmation
data, samplerate = sf.read(filename)
transcription=[""]

# Initialize Wav2Vec2 processor and model
processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Callback function to process audio data
def callback(indata, frames, time, status):
    global transcription
    if status:
        print(f"Status: {status}")
    
    # Convert audio data to numpy array
    audio = indata.flatten()

    # Check if the maximum absolute value exceeds the energy threshold to filter out silence
    if np.max(np.abs(audio)) > energy_threshold:
        # Convert audio data to tensor
        input_values = processor(audio, sampling_rate=freq, return_tensors="pt", padding=True).input_values.to(device)
        
        # Perform inference
        with torch.no_grad():
            logits = model(input_values).logits
        
        # Decode the predicted ids
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)
        print(transcription[0])

# Create an InputStream

# Assign the appropriate value to device by running: python -m sounddevice
# Device assignment isn't mandatory on windows,but it is on Raspberry Pi
stream_asr = sd.InputStream(device=1, callback=callback, channels=channels, samplerate=freq, blocksize=48000)

# Function to run YOLO and detect objects
def run_yolo():
    model = YOLO("vision.pt")
    class_names = model.names  # Get class names

    # Run inference on the source (camera feed or image)
    # Assign the appropriate value to source
    results = model.predict(source=0, imgsz=320, stream=True, verbose=False)

    for result in results:
        detected_objects = []
        for box in result.boxes:
            class_id = int(box.cls)
            detected_objects = class_names[class_id]  # Append detected class name
            
        if detected_objects:
            print(detected_objects)
            if not stream_asr.active:  # Check if it's already running
                stream_asr.start()
            if transcription[0]:
                if(detected_objects == transcription[0]):
                    sd.play(data, samplerate)
                    sd.wait()
            else:
                print("You can start talking")
                    
        time.sleep(3)  # Add a small delay between detections

# Run YOLO and Piper concurrently using threading
if __name__ == "__main__":
    run_yolo()
