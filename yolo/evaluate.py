from ultralytics import YOLO
from pathlib import Path
import argparse
import json

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=int, choices=[1, 2, 3], default=1,
                    help='Specify which dataset to evaluate on (1, 2, or 3 (combined))')
parser.add_argument('--split', type=str, choices=['val', 'test'], default='test',
                    help='Specify which split to evaluate on (val or test)')
args = parser.parse_args()

# Base paths
BASE_DIR = Path("alphabet_datasets")
RUNS_DIR = Path("runs/yolo")

# Dataset mapping
DATASET_MAP = {
    1: ("dataset1", "yolo_classifier_1.pt"),
    2: ("dataset2", "yolo_classifier_2.pt"),
    3: ("combined", "yolo_classifier_combined.pt")
}

def evaluate_model():
    # Get dataset info
    dataset_name, model_name = DATASET_MAP[args.dataset]
    dataset_path = BASE_DIR / dataset_name
    model_path = RUNS_DIR / dataset_name / "weights" / model_name
    
    # Load the trained model
    model = YOLO(str(model_path))
    
    # Evaluate on specified split
    results = model.val(data=str(dataset_path), split=args.split)
    
    # Save results
    results_dict = {
        "dataset": dataset_name,
        "split": args.split,
        "top1_accuracy": results.top1,
        "top5_accuracy": results.top5
    }
    
    # Save results to JSON
    output_dir = RUNS_DIR / dataset_name / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / f"{args.split}_results.json", "w") as f:
        json.dump(results_dict, f, indent=4)
    
    # Print results
    print(f"\n{args.split.capitalize()} Results for {dataset_name}:")
    print(f"Top-1 Accuracy: {results.top1}")
    print(f"Top-5 Accuracy: {results.top5}")
    print(f"Results saved to: {output_dir / f'{args.split}_results.json'}")

if __name__ == "__main__":
    evaluate_model() 