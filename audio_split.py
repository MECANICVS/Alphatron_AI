from pydub import AudioSegment
import os

# Function to split audio into 1-second chunks
def slice_audio(file_path, output_folder):
    # Load the audio file
    audio = AudioSegment.from_wav(file_path)
    
    # Calculate the length of the audio in milliseconds
    audio_length = len(audio)
    
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Split the audio into 1-second chunks
    for i in range(0, audio_length, 1000):  # 1000 ms = 1 second
        chunk = audio[i:i+1000]
        chunk_name = f"file_c{i//1000}.wav"
        chunk.export(os.path.join(output_folder, chunk_name), format="wav")
        print(f"Exported {chunk_name}")


input_file = "recordings\\file.wav"  # Path to your input .wav file
output_dir = "splits"    # Directory to save the chunks
slice_audio(input_file, output_dir)
