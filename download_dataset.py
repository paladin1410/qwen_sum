from datasets import load_dataset
import os

# --- Configuration ---
DATASET_NAME = "cnn_dailymail"
DATASET_VERSION = "3.0.0"
SAVE_PATH = "./offline_data/cnn_dailymail"

# --- Logic ---
print(f"Downloading {DATASET_NAME} (version {DATASET_VERSION})...")

# Create the save directory if it doesn't exist
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

# Download the entire dataset (all splits)
dataset = load_dataset(DATASET_NAME, DATASET_VERSION)

# Save it to a local directory. This format is efficient and self-contained.
print(f"Saving dataset to disk at: {SAVE_PATH}")
dataset.save_to_disk(SAVE_PATH)

print(f"✅ Download and save complete. The dataset is in the '{SAVE_PATH}' directory.")