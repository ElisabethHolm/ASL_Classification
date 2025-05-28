from ultralytics import YOLO
from pathlib import Path
import sys

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent.parent))
from yolo.config import DATASET_PATHS, MODEL_PATHS

def validate_dataset1():
    # Load the best model from training
    model = YOLO(str(MODEL_PATHS["dataset1"] / "weights" / "best.pt"))
    
    # Validate on dataset1
    metrics = model.val(data=str(DATASET_PATHS["dataset1"]))
    
    # Print results
    print("\nValidation Results for Dataset 1:")
    print(f"Top-1 Accuracy: {metrics.top1}")
    print(f"Top-5 Accuracy: {metrics.top5}")

if __name__ == "__main__":
    validate_dataset1() 