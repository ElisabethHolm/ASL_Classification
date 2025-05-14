# before running for the first time do: wget -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

import mediapipe as mp
import cv2
import os
import numpy as np
import json
from tqdm import tqdm
import kagglehub

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

# init hand detectors
model_path = "./hand_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=RunningMode.IMAGE
)

hand_landmarker = HandLandmarker.create_from_options(options)

# get keypoints from an image
def extract_keypoints_from_image(image_path):
    # read image and convert to RGB
    image = cv2.imread(image_path)
    if image is None:
        return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Wrap image in MediaPipe Image class with format (expected input to media pipe)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    # Run the detector
    result = hand_landmarker.detect(mp_image)

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        return [coord for lm in hand for coord in (lm.x, lm.y, lm.z)]

    return None

# get keypoints from all images and save in json
def process_dataset(dataset_dir="./alphabet_dataset_mp/", output_json="keypoints_dataset.json", max_samples=1000):
    data, labels = [], []
    valid_labels = [chr(ord('A') + i) for i in range(26)] + ['del', 'nothing', 'space']

    for label in sorted(os.listdir(dataset_dir)):
        if label not in valid_labels:
            continue

        label_path = os.path.join(dataset_dir, label)
        count = 0
        for file in tqdm(os.listdir(label_path), desc=f"Processing {label}"):
            if count >= max_samples:
                break
            img_path = os.path.join(label_path, file)
            keypoints = extract_keypoints_from_image(img_path)
            if keypoints:
                data.append(keypoints)
                labels.append(label)
                count += 1

    with open(output_json, "w") as f:
        json.dump({"data": data, "labels": labels}, f)
    print(f"Saved {len(data)} samples.")

if __name__ == "__main__":
    process_dataset()