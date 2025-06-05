from ultralytics import YOLO
from pathlib import Path
import argparse
import cv2
import numpy as np
import mediapipe as mp
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=int, choices=[1, 2, 3], default=1,
                    help='Specify which model to use (1, 2, or 3 (combined))')
parser.add_argument('--source', type=str, default='0',
                    help='Video source (0 for webcam, or path to video file)')
parser.add_argument("-a", "--acc_test", action="store_true", 
                    help="Run formal accuracy test mode")
args = parser.parse_args()

RUNS_DIR = Path("runs/yolo")

DATASET_MAP = {
    1: ("dataset1/train4", "best.pt"),
    2: ("dataset2/train", "best.pt"),  
    3: ("combined/train4", "best.pt")  
}

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

if os.path.exists("label_classes.json"):
    with open("label_classes.json", "r") as f:
        label_classes = json.load(f)
else:
    label_classes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", 
                    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", 
                    "del", "nothing", "space"]

def extract_hand_landmarks(frame):
    """Extract hand landmarks using MediaPipe"""
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)
    if result.multi_hand_landmarks:
        return result.multi_hand_landmarks[0]
    return None

def run_continuous_inference():
    """Run continuous real-time inference with hand visualization"""
    dataset_name, model_name = DATASET_MAP[args.dataset]
    model_path = RUNS_DIR / dataset_name / "weights" / model_name
    
    model = YOLO(str(model_path))
    
    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    print(f"Running real-time inference using {dataset_name} model")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        landmarks = extract_hand_landmarks(frame)
        
        results = model(frame)
        
        pred = results[0]
        if pred.probs is not None:
            top_prob = pred.probs.top1
            confidence = pred.probs.top1conf.item()
            class_name = pred.names[top_prob]
            
            text = f"{class_name}: {confidence:.2f}"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No Hand Detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        if landmarks:
            mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
        
        cv2.imshow('ASL Classification', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

def run_accuracy_test():
    """Run formal accuracy test for each letter"""
    dataset_name, model_name = DATASET_MAP[args.dataset]
    model_path = RUNS_DIR / dataset_name / "weights" / model_name
    
    # Load the trained model
    model = YOLO(str(model_path))
    
    # Open video capture
    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    accuracies = {}
    
    # Test each letter
    for i, testing_label in enumerate(label_classes):
        fps = 30
        seconds_per_label = 5
        num_correct = 0
        total_frames = fps * seconds_per_label
        
        print(f"\nTesting letter: {testing_label}")
        print("Get ready...")
        cv2.waitKey(2000)  
        
        for frame_num in range(total_frames):
            is_last_frame = (frame_num + 1 == total_frames)
            
            ret, frame = cap.read()
            if not ret:
                break
            
            landmarks = extract_hand_landmarks(frame)
            
            results = model(frame)
            pred = results[0]
            
            if pred.probs is not None:
                top_prob = pred.probs.top1
                confidence = pred.probs.top1conf.item()
                predicted_label = pred.names[top_prob]
                
                if predicted_label == testing_label:
                    num_correct += 1
            else:
                predicted_label = "No Hand"
            
            if landmarks:
                mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
            
            cv2.putText(frame, f"Testing: {testing_label}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Prediction: {predicted_label}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if is_last_frame and i + 1 < len(label_classes):
                cv2.putText(frame, f"Next: {label_classes[i+1]}", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, "Press any key to continue", (10, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("ASL Accuracy Test", frame)
            
            if is_last_frame:
                cv2.waitKey(0)  
            else:
                cv2.waitKey(1)
        
        accuracy = num_correct / total_frames
        accuracies[testing_label] = accuracy
        print(f"Accuracy for {testing_label}: {accuracy:.2%}")
    
    print("\nFinal Accuracy Results:")
    for letter, acc in accuracies.items():
        print(f"{letter}: {acc:.2%}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if args.acc_test:
        run_accuracy_test()
    else:
        run_continuous_inference() 