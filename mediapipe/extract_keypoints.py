# before running for the first time do: wget -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

import mediapipe as mp
import cv2
import os
import numpy as np
import json
from tqdm import tqdm

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
def process_dataset(dataset_dir="./alphabet_datasets_mp/", output_json="keypoints_dataset.json", save_every=500):
    if os.path.exists(output_json):
        print(f"Skipping {dataset_dir}: found existing {output_json}")
        return
    
    print(f"Extracting keypoints for {output_json}")
    valid_labels = [chr(ord('A') + i) for i in range(26)] + ['del', 'nothing', 'space']
    
    # Resume from existing file if it exists
    if os.path.exists(output_json):
        with open(output_json, "r") as f:
            obj = json.load(f)
            data, labels = obj["data"], obj["labels"]
    else:
        data, labels = [], []

    sample_count = len(data)

    for label in sorted(os.listdir(dataset_dir)):
        if label not in valid_labels:
            continue

        label_path = os.path.join(dataset_dir, label)
        for file in tqdm(os.listdir(label_path), desc=f"Processing {label}"):
            img_path = os.path.join(label_path, file)

            keypoints = extract_keypoints_from_image(img_path)
            if keypoints:
                data.append(keypoints)
                labels.append(label)
                sample_count += 1

                # Save periodically
                if sample_count % save_every == 0:
                    with open(output_json, "w") as f:
                        json.dump({"data": data, "labels": labels}, f)
                    print(f"Checkpoint: {sample_count} samples saved.")

    # Final save
    with open(output_json, "w") as f:
        json.dump({"data": data, "labels": labels}, f)
    print(f"Final save: {len(data)} samples.")


# merge existing jsons into a combined json
def combine_jsons(json1_path, json2_path, combined_path):
    if os.path.exists(combined_path):
        print(f"Skipping combine: {combined_path} already exists.")
        return

    print("Combining JSONs...")
    with open(json1_path, "r") as f1, open(json2_path, "r") as f2:
        data1 = json.load(f1)
        data2 = json.load(f2)

    combined_data = {
        "data": data1["data"] + data2["data"],
        "labels": data1["labels"] + data2["labels"]
    }

    with open(combined_path, "w") as f:
        json.dump(combined_data, f)
    print(f"Combined JSON written to {combined_path}")


if __name__ == "__main__":
    process_dataset(dataset_dir="./alphabet_datasets_mp/dataset1", output_json="keypoints_d1.json")

    process_dataset(dataset_dir="./alphabet_datasets_mp/dataset2", output_json="keypoints_d2.json")

    combine_jsons("keypoints_d1.json", "keypoints_d2.json", "keypoints_combined.json")