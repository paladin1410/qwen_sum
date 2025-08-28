from huggingface_hub import snapshot_download
import os

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
SAVE_PATH = "./models/qwen-model"

# --- Logic ---
print(f"Downloading {MODEL_NAME}...")

# Create the save directory if it doesn't exist
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

# Download the model
snapshot_download(
    repo_id=MODEL_NAME,
    local_dir=SAVE_PATH,
    local_dir_use_symlinks=False
)

print(f"Download complete. The model is in the '{SAVE_PATH}' directory.")