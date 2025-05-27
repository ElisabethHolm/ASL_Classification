import kagglehub
import os
import shutil
from pathlib import Path

def copy_dataset(source_dir, target_dir):
    for class_folder in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_folder)
        target_class_path = os.path.join(target_dir, class_folder)
        if os.path.isdir(class_path):
            shutil.copytree(class_path, target_class_path, dirs_exist_ok=True)

def copy_with_rename(src_dir, dst_dir, prefix):
    for img in Path(src_dir).glob("*"):
        if img.is_file():
            new_name = f"{prefix}_{img.name}"
            shutil.copy(img, dst_dir / img.parent.name / new_name)

def combine_datasets(d1_dir, d2_dir, combined_dir):
    d1_dir = Path(d1_dir)
    d2_dir = Path(d2_dir)
    combined_dir = Path(combined_dir)
    
    for cls in sorted(os.listdir(d1_dir)):
        (combined_dir / cls).mkdir(parents=True, exist_ok=True)

    for cls in sorted(os.listdir(d2_dir)):
        (combined_dir / cls).mkdir(parents=True, exist_ok=True)

    for cls in sorted(os.listdir(d1_dir)):
        copy_with_rename(d1_dir / cls, combined_dir, "d1")
    for cls in sorted(os.listdir(d2_dir)):
        copy_with_rename(d2_dir / cls, combined_dir, "d2")

# Base output directory
base_dir = Path("./alphabet_datasets_mp")
dataset1_dir = base_dir / "dataset1"
dataset2_dir = base_dir / "dataset2"
combined_dir = base_dir / "combined"

base_dir.mkdir(exist_ok=True)

#  Dataset 1
if not dataset1_dir.exists():
    print("Downloading Dataset 1 from KaggleHub...")
    d1_path = kagglehub.dataset_download("grassknoted/asl-alphabet")
    src1 = Path(d1_path) / "asl_alphabet_train" / "asl_alphabet_train"
    dataset1_dir.mkdir(parents=True)
    copy_dataset(src1, dataset1_dir)
    print("Dataset 1 downloaded and copied.")
else:
    print(f"Dataset 1 already exists at {dataset1_dir}, skipping download.")

# Dataset 2 
if not dataset2_dir.exists():
    print("Downloading Dataset 2 from KaggleHub...")
    d2_path = kagglehub.dataset_download("debashishsau/aslamerican-sign-language-aplhabet-dataset")
    src2 = Path(d2_path) / "ASL_Alphabet_Dataset" / "asl_alphabet_train"
    dataset2_dir.mkdir(parents=True)
    copy_dataset(src2, dataset2_dir)
    print("Dataset 2 downloaded and copied.")
else:
    print(f"Dataset 2 already exists at {dataset2_dir}, skipping download.")

# Combine
print("Combining datasets...")
combine_datasets(dataset1_dir, dataset2_dir, combined_dir)
print("Combined dataset created.")