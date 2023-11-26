import argparse
from pathlib import Path
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import os
from transformers import pipeline
import shutil
from PIL import Image
from clearml import Task

parser = argparse.ArgumentParser(description="Mass score a set of images")

# Images arguments
parser.add_argument(
    "--input-folder",
#    required=True,
    help="Folder with the images to score",
)

# Model arguments
parser.add_argument("--model-folder", default="shadowlilac/aesthetic-shadow")
parser.add_argument(
    "--batch-hq-threshold",
    default=0.5,
    type=float,
    help="Predictions threshold",
)
parser.add_argument(
    "--batch-size",
    default=8,
    type=int,
    help="Batch size",
)
args = parser.parse_args()
input_folder = args.input_folder

model_folder = args.model_folder
batch_hq_threshold = args.batch_hq_threshold
batch_size = args.batch_size

pipe = pipeline("image-classification", model=model_folder, device=0)
# Define the paths for the input folder and output folders

output_folder_hq = input_folder+"output_hq_folder" 
if not os.path.exists(output_folder_hq):
   # Create a new directory because it does not exist
   os.makedirs(output_folder_hq)
output_folder_lq = input_folder+"output_lq_folder" 
if not os.path.exists(output_folder_lq):
   # Create a new directory because it does not exist
   os.makedirs(output_folder_lq)
# List all image files in the input folder
image_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]


# Process images in batches
for i in range(0, len(image_files), batch_size):
    batch = image_files[i:i + batch_size]

    # Perform classification for the batch
    results = pipe(images=batch)

    for idx, result in enumerate(results):
        # Extract the prediction scores and labels
        predictions = result
        hq_score = [p for p in predictions if p['label'] == 'hq'][0]['score']

        # Determine the destination folder based on the prediction and threshold
        destination_folder = output_folder_hq if hq_score >= batch_hq_threshold else output_folder_lq

        # Copy the image to the appropriate folder
        shutil.copy(batch[idx], os.path.join(destination_folder, os.path.basename(batch[idx])))

print("Classification and sorting complete.")
