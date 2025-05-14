import kagglehub
import os

# download kaggle dataset if not already
target_dir = "./alphabet_dataset_mp"

if not os.path.exists(target_dir):
    # get data from kaggle
    print("Downloading ASL Alphabet dataset from KaggleHub...")
    dataset_path = kagglehub.dataset_download("grassknoted/asl-alphabet")
    source_dir = os.path.join(dataset_path, "asl_alphabet_train/asl_alphabet_train")
    
    # make new dir for it
    os.makedirs(target_dir)
    
    import shutil
    # Copy each class subfolder individually
    for class_folder in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_folder)
        target_class_path = os.path.join(target_dir, class_folder)
        if os.path.isdir(class_path):
            shutil.copytree(class_path, target_class_path)
    print("Download and copy complete.")
else:
    print(f"Dataset already exists at {target_dir}, skipping download.")