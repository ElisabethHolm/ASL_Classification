from pathlib import Path

# Base paths
BASE_DIR = Path("alphabet_datasets")
RUNS_DIR = Path("runs/yolo")

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

# Dataset paths
DATASET_PATHS = {
    "dataset1": BASE_DIR / "dataset1",
    "dataset2": BASE_DIR / "dataset2",
    "combined": BASE_DIR / "combined"
}

# Model paths
MODEL_PATHS = {
    "dataset1": RUNS_DIR / "dataset1" / "train",
    "dataset2": RUNS_DIR / "dataset2" / "train",
    "combined": RUNS_DIR / "combined" / "train"
} 