# directory structure for compatibility with YoloV11
# https://docs.ultralytics.com/datasets/classify/
import kagglehub
import os
import shutil
import random
from pathlib import Path

# Download dataset
path = kagglehub.dataset_download("grassknoted/asl-alphabet")
print("Path to original dataset files:", path)

# Define input and output paths
train_dir = Path(path) / "asl_alphabet_train" / "asl_alphabet_train"
output_dir = Path("alphabet_dataset")
train_output = output_dir / "train"
val_output = output_dir / "val"

# Make output directories
for d in [train_output, val_output]:
    d.mkdir(parents=True, exist_ok=True)

# Validation split ratio
VAL_RATIO = 0.2

# Process each class folder
classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
for cls in classes:
    src_dir = train_dir / cls
    images = list(src_dir.glob("*.jpg"))
    random.shuffle(images)
    split_idx = int(len(images) * (1 - VAL_RATIO))
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    # Create class subfolders in output
    cls_train_dir = train_output / cls
    cls_val_dir = val_output / cls
    cls_train_dir.mkdir(parents=True, exist_ok=True)
    cls_val_dir.mkdir(parents=True, exist_ok=True)

    # Copy images
    for img in train_imgs:
        shutil.copy(img, cls_train_dir / img.name)
    for img in val_imgs:
        shutil.copy(img, cls_val_dir / img.name)

print(f"Classification dataset created at: {output_dir.resolve()}")