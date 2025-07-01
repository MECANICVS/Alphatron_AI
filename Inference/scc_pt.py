"""This script allows you to use different words to run other AI models,or perform any other action"""
#example usage: python scc_pt.py --model model.pth --submodel yoloxpiper.py --sdev 1 --labelmap labelmap.txt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
# On Raspberry Pi, you may have to run "sudo apt install libsox-dev libsndfile1" for torchaudio
import torchaudio.transforms as T
import sounddevice as sd
# On Raspberry Pi, additionally to "pip install sounddevice", type "sudo apt install portaudio19-dev"
import soundfile as sf
import numpy as np
import threading
import argparse
import os
import random
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import subprocess
from torch.quantization import quantize_dynamic

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Speech Command Model', required=True)
parser.add_argument('--submodel', help='The other AI Model, yolo_detect.py|yoloxpiper.py|yoloxwav2vec.py ', required=True)
parser.add_argument('--sdev', help='Sound Device Index, you can get it by running, python -m sounddevice', required=True)
parser.add_argument('--labelmap', help='The labelmap', required=True)

args = parser.parse_args()

# Parse user inputs
model_path = args.model
script_path = args.submodel
sdevice = args.sdev
labels = args.labelmap


# Load labels from file
def load_labels(file_path):
    label_dict = {}
    
    try:
        with open(file_path, "r") as f:
            for line in f:
                index, label = line.strip().split(",")
                label_dict[int(index)] = label
    except Exception as e:
        print(f"Error loading labels: {e}")
        return None
    return label_dict

label_map = load_labels(labels)
num_classes = len(label_map)

# Speech Command Model Architecture
class SpeechCommandCNN(nn.Module):
    def __init__(self, n_channels=1, n_classes=10, n_filters=32):
        super().__init__()

        self.conv1 = nn.Conv2d(n_channels, n_filters, kernel_size=(3, 3), padding=(1, 1), bias=False)  # Kernel for time & frequency
        self.bn1 = nn.BatchNorm2d(n_filters)      

        self.conv2 = nn.Conv2d(n_filters, n_filters, kernel_size=(3, 3), padding=(1, 1), bias=False)  # 3x3 kernel to capture local features
        self.bn2 = nn.BatchNorm2d(n_filters)
        self.pool1 = nn.MaxPool2d((2, 2))      

        self.conv3 = nn.Conv2d(n_filters, 2 * n_filters, kernel_size=(3, 3), padding=(1, 1), bias=False)  # Larger kernel to capture more complex features
        self.bn3 = nn.BatchNorm2d(2 * n_filters)      

        self.conv4 = nn.Conv2d(2 * n_filters, 2 * n_filters, kernel_size=(3, 3), padding=(1, 1), bias=False)  # Deeper convolution
        self.bn4 = nn.BatchNorm2d(2 * n_filters)
        self.pool2 = nn.MaxPool2d((2, 2))
        
        self.fc1 = nn.Linear(2 * n_filters, 128)
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))

        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool1(x)
        
        x = self.conv3(x)
        x = F.relu(self.bn3(x))

        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool2(x)

        x = F.avg_pool2d(x, (x.shape[-2], x.shape[-1]))  # Global average pooling

        x = x.view(x.size(0), -1)  # Flatten to [batch_size, filters]

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")# CPU only if quantized model
best_model = SpeechCommandCNN(n_classes=num_classes)
best_model.load_state_dict(torch.load(model_path, map_location=device))
best_model.to(device)
best_model.eval()
# Dynamic Quantization, comment if not needed
quantized_model = torch.quantization.quantize_dynamic(
    best_model, 
    {torch.nn.Linear},
    dtype=torch.qint8
)
quantized_model.eval()
# print(quantized_model)


# Audio parameters
SAMPLE_RATE = 16000
CHANNELS = 1
BUFFER_SIZE = 16000
audio_buffer = []
predicted_index = None
process = None
arguments = ["--model", "vision.pt", "--source", "usb0"] #arguments for yolo_detect.py


# When audio files are in a subfolder (or not at the same level as the inference script), use absolute path
audio_dir = 'C:\\Users\\username\\anaconda3\\envs\\pyt\\audio_files'
all_files = os.listdir(audio_dir)
audio_files = [file for file in all_files if file.endswith((".wav", ".mp3"))]
# when the audio and the script are at the same level, you can use relative path
filename1 = "cv.wav"
data1, samplerate1 = sf.read(filename1)

# Confidence & Entropy Calculation
def calculate_confidence_and_entropy(output):
    probabilities = F.softmax(output, dim=1)
    confidence = probabilities.max().item()
    entropy_score = -torch.sum(probabilities * torch.log(probabilities + 1e-8), dim=1)
    return confidence, entropy_score.item()
    
def preprocess_live_audio(audio_tensor):
    """Convert live audio tensor to Mel-Spectrogram format."""

    audio_length = audio_tensor.size(1)

    transforms = torch.nn.Sequential(
        MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=64, n_fft=2048, hop_length=400),
        AmplitudeToDB()
    )

    mel_spec = transforms(audio_tensor)  # Shape: [1, 64, time_bins]

    mel_spec = mel_spec.unsqueeze(0)  # Ensure batch dimension [1, 1, H, W]
    
#    print(f"Live processed shape: {mel_spec.shape}")
    return mel_spec
    
# Callback for real-time audio input
def audio_callback(indata, frames, time, status):
    global predicted_index, audio_buffer
    if status:
        print(f"Status: {status}")
    audio_chunk = np.array(indata).flatten()
    # Check if there is speech (simple energy threshold to filter out silence)
    if np.max(np.abs(audio_chunk)) > 0.02:  #recommend between 0.01 and 0.05

        audio_buffer.append(audio_chunk)
        accumulated_audio = np.concatenate(audio_buffer, axis=0)

        if len(accumulated_audio) >= BUFFER_SIZE:
            audio_tensor = torch.tensor(accumulated_audio).float().unsqueeze(0)
            process_audio(audio_tensor)
            audio_buffer.clear()

def process_audio(audio_tensor):
    """Process the audio tensor to detect speech and make predictions."""
    global predicted_index, process
    
    processed_audio = preprocess_live_audio(audio_tensor).to(device)
#    print(f"Processed audio shape: {processed_audio.shape}")
    
    # Run inference with the model (regular or quantized)
    with torch.no_grad():
        output = best_model(processed_audio)
 #       output = quantized_model(processed_audio)
        predicted_index = output.argmax(dim=1).item()
        confidence, entropy = calculate_confidence_and_entropy(output)

        if confidence > CONFIDENCE_THRESHOLD and entropy < ENTROPY_THRESHOLD:
            print(f"{label_map.get(predicted_index, 'Unknown')} "
                  f"(Confidence: {confidence:.2f}, Entropy: {entropy:.2f})")
            if predicted_index == 0:
                filename = random.choice(audio_files)# select a random audio in the folder
                full_path = os.path.join(audio_dir, filename)
                data, samplerate = sf.read(full_path)
                sd.play(data, samplerate)
                sd.wait() # Wait until playback is finished              
            elif predicted_index == 7 and process is None:
                sd.play(data1, samplerate1)
                sd.wait()
                process = subprocess.Popen(["python", script_path] + arguments, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
            elif predicted_index == 6 and process is not None:
                process.terminate()                
                stdout, stderr = process.communicate()  # Collect output after termination
                print("Computer Vision terminated.")
                print("STDOUT:", stdout)
                print("STDERR:", stderr)
                process = None
        else: 
            print(f"UNCERTAIN: {label_map.get(predicted_index, 'Unknown')} "
                  f"(Confidence: {confidence:.2f}, Entropy: {entropy:.2f})")
           
            
# Real-time audio classification with uncertainty management
CONFIDENCE_THRESHOLD = 0.7
ENTROPY_THRESHOLD = 0.4
# Device assignment isn't mandatory on windows, but it is on Raspberry Pi
stream = sd.InputStream(device=int(sdevice), samplerate=SAMPLE_RATE, channels=CHANNELS, callback=audio_callback, blocksize=BUFFER_SIZE)
stream.start()
print("Streaming started. You can stop it manually (ctrl+c)")
try:
    while stream.active:
        pass
except KeyboardInterrupt:
    print("Audio Stream Stopped")
    stream.stop()
