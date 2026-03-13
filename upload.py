import os
from huggingface_hub import HfApi, create_repo, login

# ========== CONFIG ==========
HF_USERNAME = "your_username"
REPO_NAME = "measles-outbreak-model"
METHOD_NAME = "RFE"   # Folder inside saved_models/
LOCAL_MODEL_DIR = f"saved_models/{METHOD_NAME}"
PRIVATE = False
# ============================

repo_id = f"{HF_USERNAME}/{REPO_NAME}"

# Login (will ask for your token)
login()

# Create repo if it doesn't exist
create_repo(repo_id, private=PRIVATE, exist_ok=True)

api = HfApi()

# Upload entire folder
print(f"\nUploading models from {LOCAL_MODEL_DIR}...\n")

api.upload_folder(
    folder_path=LOCAL_MODEL_DIR,
    repo_id=repo_id,
    repo_type="model"
)

print(" Models uploaded successfully!")

# -------- Optional: Create README automatically --------
readme_content = f"""
# Measles Outbreak Prediction Model

This repository contains stacked deep learning models trained using the {METHOD_NAME} feature selection method.

## Models Included
- LSTM
- BiLSTM
- GRU
- Logistic Regression Meta Learner

## Task
Binary classification (0 = Non-Outbreak, 1 = Outbreak)

## Frameworks
- TensorFlow / Keras
- Scikit-learn
"""

with open("README.md", "w") as f:
    f.write(readme_content)

api.upload_file(
    path_or_fileobj="README.md",
    path_in_repo="README.md",
    repo_id=repo_id,
)

print(" README uploaded!")

# -------- Optional: requirements.txt --------
requirements = """tensorflow
scikit-learn
numpy
pandas
joblib
"""

with open("requirements.txt", "w") as f:
    f.write(requirements)

api.upload_file(
    path_or_fileobj="requirements.txt",
    path_in_repo="requirements.txt",
    repo_id=repo_id,
)

print(" requirements.txt uploaded!")
print(f"\n All done! View your model at:")
print(f"https://huggingface.co/{repo_id}")
