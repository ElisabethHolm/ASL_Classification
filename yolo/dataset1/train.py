from ultralytics import YOLO
from pathlib import Path
import sys

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent.parent))
from yolo.config import COMMON_PARAMS, DATASET_PATHS, MODEL_PATHS

def train_dataset1():
    # Load pretrained model
    model = YOLO("yolo11n-cls.pt")
    
    # Train on dataset1
    results = model.train(
        data=str(DATASET_PATHS["dataset1"]),
        project=str(MODEL_PATHS["dataset1"].parent),
        name="train",
        **COMMON_PARAMS
    )
    
    # Print results
    print("\nTraining Results for Dataset 1:")
    print(f"Top-1 Accuracy: {results.top1}")
    print(f"Top-5 Accuracy: {results.top5}")

if __name__ == "__main__":
    train_dataset1() 