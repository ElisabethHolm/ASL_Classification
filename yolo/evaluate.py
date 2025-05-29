from ultralytics import YOLO
from pathlib import Path
import argparse
import json
import torch

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=int, choices=[1, 2, 3], default=1,
                    help='Specify which dataset to evaluate on (1, 2, or 3 (combined))')
parser.add_argument('--split', type=str, choices=['val', 'test'], default='val',
                    help='Specify which split to evaluate on (val or test)')
args = parser.parse_args()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Base paths
BASE_DIR = Path(__file__).parent.parent / "alphabet_datasets"
RUNS_DIR = Path(__file__).parent.parent / "runs/yolo"

# Dataset mapping
DATASET_MAP = {
    1: ("dataset1", "train4"),  # Using the most recent training run
    2: ("dataset2", "train"),   # Using the train directory for dataset2
    3: ("combined", "train")    # Will use train directory for combined when trained
}

def evaluate_model():
    # Get dataset info
    dataset_name, train_run = DATASET_MAP[args.dataset]
    dataset_path = BASE_DIR / dataset_name
    model_path = RUNS_DIR / dataset_name / train_run / "weights" / "best.pt"
    
    # Verify dataset and model exist
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    print(f"Evaluating on dataset: {dataset_name}")
    print(f"Dataset path: {dataset_path}")
    print(f"Model path: {model_path}")
    print(f"Split: {args.split}")
    print(f"Using device: {device}")
    
    # Load model
    model = YOLO(str(model_path))
    
    # Evaluate the model
    results = model.val(
        data=str(dataset_path),
        split=args.split,
        device=device
    )
    
    # Save results
    results_dict = {
        "dataset": dataset_name,
        "split": args.split,
        "top1_accuracy": results.top1,
        "top5_accuracy": results.top5,
        "device": str(device)
    }
    
    # Save results to JSON
    results_path = RUNS_DIR / dataset_name / f"{args.split}_results.json"
    with open(results_path, "w") as f:
        json.dump(results_dict, f, indent=4)
    
    # Print results
    print(f"\nEvaluation Results for {dataset_name} ({args.split}):")
    print(f"Top-1 Accuracy: {results.top1}")
    print(f"Top-5 Accuracy: {results.top5}")
    print(f"Results saved to: {results_path}")

if __name__ == "__main__":
    evaluate_model() 