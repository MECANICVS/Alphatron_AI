"""This script is to run inference with the converted onnx model"""
import numpy as np
import sounddevice as sd
import onnx
import onnxruntime as ort
import librosa

# Audio parameters
SAMPLE_RATE = 16000
CHANNELS = 1
BUFFER_SIZE = 16000
audio_buffer = []

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

label_map = load_labels("labelmap.txt")

# Confidence & Entropy Calculation
def calculate_confidence_and_entropy(output):
    probabilities = np.exp(output) / np.sum(np.exp(output), axis=1, keepdims=True)  # Softmax for ONNX output
    confidence = np.max(probabilities)
    entropy_score = -np.sum(probabilities * np.log(probabilities + 1e-8), axis=1)
    return confidence, entropy_score.item()

# ONNX Inference Model
class ONNXModel:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'], sess_options=ort.SessionOptions())
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, input_data):
        input_data = input_data.astype(np.float32)  # Ensure float32 type
        result = self.session.run([self.output_name], {self.input_name: input_data})
        return result[0]

# Load ONNX model
onnx_model = ONNXModel("model.onnx")#Make sure model.onnx and model.onnx.data are in the same folder

def preprocess_live_audio(audio_tensor):
    """Convert live audio tensor to Mel-Spectrogram format (manual conversion)."""
    # Calculate dynamic values for n_fft and hop_length based on audio length
    audio_length = audio_tensor.shape[1]

    # Manually convert to Mel-Spectrogram using librosa
    mel_spec = librosa.feature.melspectrogram(y=audio_tensor.flatten(), sr=SAMPLE_RATE, n_mels=64, n_fft=2048, hop_length=400)
    mel_spec_db = librosa.power_to_db(mel_spec)

    mel_spec_db = np.expand_dims(mel_spec_db, axis=0)  # Adding batch dimension
    mel_spec_db = np.expand_dims(mel_spec_db, axis=0)  # Adding channel dimension (1)

#    print(f"Processed mel-spec shape: {mel_spec_db.shape}")
    return mel_spec_db

# Callback for real-time audio input
def audio_callback(indata, frames, time, status):
    if status:
        print(f"Status: {status}")

    audio_chunk = np.array(indata).flatten()

    # Check if there is speech (simple energy threshold)
    if np.max(np.abs(audio_chunk)) > 0.015:  # Adjust threshold experimentally
        audio_buffer.append(audio_chunk)
        accumulated_audio = np.concatenate(audio_buffer, axis=0)

        if len(accumulated_audio) >= BUFFER_SIZE:
            audio_tensor = np.array(accumulated_audio).astype(np.float32).reshape(1, -1)
            processed_audio = preprocess_live_audio(audio_tensor)
#            print(f"Processed live audio shape: {processed_audio.shape}")

            # Run inference using ONNX model
            output = onnx_model.predict(processed_audio)
            predicted_index = np.argmax(output)  # Extract the class index
            confidence, entropy_score = calculate_confidence_and_entropy(output)
            # Map the index to the corresponding label
            predicted_label = label_map.get(predicted_index, "Unknown")
            
            if confidence > CONFIDENCE_THRESHOLD or entropy_score < ENTROPY_THRESHOLD:
                print(f"Predicted: {predicted_label} (Confidence: {confidence:.2f}, Entropy: {entropy_score:.2f})")
            else:
                print("UNCERTAIN, REPEAT")
            
            audio_buffer.clear()

# Real-time audio classification with uncertainty management
CONFIDENCE_THRESHOLD = 0.7
ENTROPY_THRESHOLD = 0.4

stream = sd.InputStream(device=1, samplerate=SAMPLE_RATE, channels=CHANNELS, callback=audio_callback, blocksize=BUFFER_SIZE)
stream.start()
print("Streaming started. You can stop it manually.")
try:
    while stream.active:
        pass
except KeyboardInterrupt:
    print("Audio Stream Stopped")
    stream.stop()
