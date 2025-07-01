import os
import csv

dataset_dir = "dataset"
output_file = "labels.csv"

with open(output_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["file name", "label"])

    for filename in os.listdir(dataset_dir):
        if filename.endswith(".wav"):
            # Extract label from filename (e.g., "start_01.wav" -> "start")
            label = filename.split("_")[0]
            writer.writerow([filename, label])