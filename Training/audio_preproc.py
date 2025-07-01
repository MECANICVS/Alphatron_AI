"""Script that resamples the audio, defines a specific length and changes stereo to mono"""
import os
import torchaudio
import torch.nn.functional as F

input_folder = "raw_dataset"
output_folder = "dataset"
target_sr = 16000  # Target sample rate or frequency
target_ns = 16000  # Target number of samples
# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)


def resample_audio(input_path, output_path, target_sr, target_ns):

    waveform, sample_rate = torchaudio.load(input_path)
    
    if sample_rate != target_sr:
#        print(f"Resampling {input_path} from {sample_rate} to (not {target_sr})Hz")
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
        waveform = resampler(waveform)
        
# Option1/ Skip files that are outside the acceptable sample length range
    if waveform.shape[-1] != target_ns:
        print(f"Skipping {input_path}, {waveform.shape[-1]}")
        return False # Skip this file 
        
# Option2/ Pad or Truncate to ensure the waveform is of the correct length        
#    if waveform.shape[-1] < target_ns:
#        waveform = F.pad(waveform, (0, target_ns - waveform.shape[-1]), "constant", 0)
#    elif waveform.shape[-1] > target_ns:
#        waveform = waveform[:, :target_ns] 
 
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # Convert to mono by averaging channel
        
    # Save the resampled waveform
    torchaudio.save(output_path, waveform, target_sr)
 #   print(f"Resampled and saved: {output_path}")
    return True



valid_files = 0
skipped_files = 0

for filename in os.listdir(input_folder):
    if filename.endswith(".wav"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        result = resample_audio(input_path, output_path, target_sr, target_ns)
        if result:
            valid_files += 1
        else:
            skipped_files += 1

print(f"Processed {valid_files} valid files and skipped {skipped_files} files.")
