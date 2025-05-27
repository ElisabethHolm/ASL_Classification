import kagglehub
import os
import shutil
import random
from pathlib import Path

# Set split ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

assert abs((TRAIN_RATIO + VAL_RATIO + TEST_RATIO) - 1.0) < 1e-6, "Splits must sum to 1.0"

# Gather all images in a directory
def gather_images(src_dir):
    return list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.jpeg"))

# Split and copy images from original to new folder
def prepare_dataset(dataset_path, dataset_name, output_base):
    train_dir = Path(dataset_path)

    # Set training data directory for dataset 1
    if dataset_name == "dataset1":
        train_dir = train_dir / "asl_alphabet_train" / "asl_alphabet_train"

    # Set training data directory for dataset 2
    elif dataset_name == "dataset2":
        train_dir = train_dir / "ASL_Alphabet_Dataset" / "asl_alphabet_train"
        print(train_dir)

    output_dir = output_base / dataset_name
    train_output = output_dir / "train"
    val_output = output_dir / "val"
    test_output = output_dir / "test"

    for d in [train_output, val_output, test_output]:
        d.mkdir(parents=True, exist_ok=True)

    classes = sorted(os.listdir(train_dir))
    print(f"Gathered the following classes in {dataset_name}: {classes}")

    for cls in classes:
        src_dir = train_dir / cls
        if not src_dir.exists():
            continue
        images = gather_images(src_dir)
        random.shuffle(images)

        n_total = len(images)
        n_train = int(n_total * TRAIN_RATIO)
        n_val = int(n_total * VAL_RATIO)

        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]

        for subset, img_list in zip(
            [train_output, val_output, test_output],
            [train_imgs, val_imgs, test_imgs]
        ):
            cls_dir = subset / cls
            cls_dir.mkdir(parents=True, exist_ok=True)
            for img in img_list:
                shutil.copy(img, cls_dir / img.name)

    print(f"{dataset_name} dataset prepared at: {output_dir.resolve()}")
    return train_output, val_output, test_output

# Output root directory
output_base = Path("alphabet_datasets")

# Download datasets
path1 = kagglehub.dataset_download("grassknoted/asl-alphabet")
print("Downloaded Dataset 1: grassknoted/asl-alphabet")

path2 = kagglehub.dataset_download("debashishsau/aslamerican-sign-language-aplhabet-dataset")
print("Downloaded Dataset 2: debashishsau/aslamerican-sign-language-aplhabet-dataset")

# Prepare Dataset 1
train1, val1, test1 = prepare_dataset(path1, "dataset1", output_base)

# Prepare Dataset 2
train2, val2, test2 = prepare_dataset(path2, "dataset2", output_base)

# Prepare Combined Dataset
combined_train_dir = output_base / "combined" / "train"
combined_val_dir = output_base / "combined" / "val"
combined_test_dir = output_base / "combined" / "test"

for d in [combined_train_dir, combined_val_dir, combined_test_dir]:
    d.mkdir(parents=True, exist_ok=True)

# Rename image with prefix so duplicate image names don't matter
def copy_with_rename(src_dir, dst_dir, prefix):
    for img in src_dir.glob("*"):
        if img.is_file():
            new_name = f"{prefix}_{img.name}"
            shutil.copy(img, dst_dir / new_name)

# Combine datasets 1 and 2
def combine_datasets(datasets, combined_output):
    splits = ["train", "val", "test"]
    for split in splits:
        for cls in sorted(os.listdir(datasets[0] / split)):
            combined_cls_dir = combined_output / split / cls
            combined_cls_dir.mkdir(parents=True, exist_ok=True)

            for i, dataset in enumerate(datasets):
                class_dir = dataset / split / cls
                if class_dir.exists():
                    copy_with_rename(class_dir, combined_cls_dir, f"d{i+1}")

combined_output = output_base / "combined"
combine_datasets(
    [output_base / "dataset1", output_base / "dataset2"],
    combined_output
)

print(f"Combined dataset created at: {output_base / 'combined'}")