from ultralytics import YOLO
from pathlib import Path
import sys
import json

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent.parent))
from yolo.config import DATASET_PATHS, MODEL_PATHS

def test_dataset1():
    # Load the best model from training
    model = YOLO(str(MODEL_PATHS["dataset1"] / "weights" / "best.pt"))
    
    # Test on dataset1 test set
    results = model.val(data=str(DATASET_PATHS["dataset1"]), split="test")
    
    # Save results to JSON
    output_dir = MODEL_PATHS["dataset1"].parent / "test"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_dict = {
        "top1_accuracy": results.top1,
        "top5_accuracy": results.top5,
        "dataset": "dataset1"
    }
    
    with open(output_dir / "test_results.json", "w") as f:
        json.dump(results_dict, f, indent=4)
    
    # Print results
    print("\nTest Results for Dataset 1:")
    print(f"Top-1 Accuracy: {results.top1}")
    print(f"Top-5 Accuracy: {results.top5}")
    print(f"Results saved to: {output_dir / 'test_results.json'}")

if __name__ == "__main__":
    test_dataset1() 