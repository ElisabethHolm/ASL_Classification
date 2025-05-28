from ultralytics import YOLO
from pathlib import Path
import argparse
import cv2
import numpy as np

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=int, choices=[1, 2, 3], default=1,
                    help='Specify which model to use (1, 2, or 3 (combined))')
parser.add_argument('--source', type=str, default='0',
                    help='Video source (0 for webcam, or path to video file)')
args = parser.parse_args()

# Base paths
RUNS_DIR = Path("runs/yolo")

# Dataset mapping
DATASET_MAP = {
    1: ("dataset1", "yolo_classifier_1.pt"),
    2: ("dataset2", "yolo_classifier_2.pt"),
    3: ("combined", "yolo_classifier_combined.pt")
}

def run_realtime_inference():
    # Get model info
    dataset_name, model_name = DATASET_MAP[args.dataset]
    model_path = RUNS_DIR / dataset_name / "weights" / model_name
    
    # Load the trained model
    model = YOLO(str(model_path))
    
    # Open video capture
    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    print(f"Running real-time inference using {dataset_name} model")
    print("Press 'q' to quit")
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run inference
        results = model(frame)
        
        # Get prediction
        pred = results[0]
        if pred.probs is not None:
            # Get top prediction
            top_prob = pred.probs.top1
            confidence = pred.probs.top1conf.item()
            class_name = pred.names[top_prob]
            
            # Display prediction
            text = f"{class_name}: {confidence:.2f}"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow('ASL Classification', frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_realtime_inference() 