from ultralytics import YOLO
from pathlib import Path
import argparse
import json

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=int, choices=[1, 2, 3], default=1,
                    help='Specify which dataset to train on (1, 2, or 3 (combined))')
args = parser.parse_args()

# Common training parameters
COMMON_PARAMS = {
    "epochs": 20,
    "imgsz": 224,
    "batch": 32,
    "workers": 4,
    "device": "cpu",  # Change to "cuda" if GPU available
    "pretrained": True,
    "optimizer": "auto",
    "verbose": True,
    "seed": 42
}

# Base paths
BASE_DIR = Path("alphabet_datasets")
RUNS_DIR = Path("runs/yolo")

# Dataset mapping
DATASET_MAP = {
    1: ("dataset1", "yolo_classifier_1.pt"),
    2: ("dataset2", "yolo_classifier_2.pt"),
    3: ("combined", "yolo_classifier_combined.pt")
}

def train_model():
    # Get dataset info
    dataset_name, model_name = DATASET_MAP[args.dataset]
    dataset_path = BASE_DIR / dataset_name
    model_path = RUNS_DIR / dataset_name / "weights" / model_name
    
    # Create output directory
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load pretrained model
    model = YOLO("yolo11n-cls.pt")
    
    # Train the model
    results = model.train(
        data=str(dataset_path),
        project=str(RUNS_DIR / dataset_name),
        name="train",
        **COMMON_PARAMS
    )
    
    # Save results
    results_dict = {
        "dataset": dataset_name,
        "top1_accuracy": results.top1,
        "top5_accuracy": results.top5,
        "epochs": COMMON_PARAMS["epochs"],
        "batch_size": COMMON_PARAMS["batch"],
        "image_size": COMMON_PARAMS["imgsz"]
    }
    
    # Save results to JSON
    with open(RUNS_DIR / dataset_name / "train_results.json", "w") as f:
        json.dump(results_dict, f, indent=4)
    
    # Print results
    print(f"\nTraining Results for {dataset_name}:")
    print(f"Top-1 Accuracy: {results.top1}")
    print(f"Top-5 Accuracy: {results.top5}")
    print(f"Results saved to: {RUNS_DIR / dataset_name / 'train_results.json'}")

if __name__ == "__main__":
    train_model() 