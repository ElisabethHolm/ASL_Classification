import argparse
import cv2
import torch
import torch.nn as nn
import mediapipe as mp
import numpy as np
import json
import os

parser = argparse.ArgumentParser(
    prog='ASLClassifier',
    description='Real-time ASL classifier using MediaPipe and a trained PyTorch model',
    epilog='Press q to quit the webcam stream.'
)
parser.add_argument('--training_data', type=int, choices=[1, 2, 3], default=1,
                    help='Specify which training dataset/model to use (1, 2, or 3 (combined dataset))')
parser.add_argument("-a", "--acc_test", action="store_true", help="If running formal real world accuracy test")
args = parser.parse_args()

# use generated json if it exists, otherwise use hard coded labels
if os.path.exists("label_classes.json"):
    with open("label_classes.json", "r") as f:
        label_classes = json.load(f)
else:
    label_classes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"]

# pytorch classifier that matches structure of trained model
class ASLClassifier(nn.Module):
    def __init__(self, input_dim=63, num_classes=len(label_classes)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)
    
# load model on device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ASLClassifier()
# load correct trained weights
if args.training_data == 1:
    model.load_state_dict(torch.load("mp_alphabet_classifier_1.pth", map_location=device))
elif args.training_data == 2:
    model.load_state_dict(torch.load("mp_alphabet_classifier_2.pth", map_location=device))
else:
    model.load_state_dict(torch.load("mp_alphabet_classifier_combined.pth", map_location=device))
model.eval().to(device)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

def extract_keypoints_from_frame(frame):
    image = cv2.resize(frame, (256, 256))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)
    if result.multi_hand_landmarks:
        lm = result.multi_hand_landmarks[0]
        keypoints = [coord for pt in lm.landmark for coord in (pt.x, pt.y, pt.z)]
        return keypoints, lm
    return None, None

# run realtime inference until user stops with 'q' key press
def run_continuous():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        keypoints, landmarks = extract_keypoints_from_frame(frame)

        if keypoints:
            x = torch.tensor(keypoints, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(x)
                _, pred = torch.max(outputs, 1)
                label = label_classes[pred.item()]
        else:
            label = "No Hand"

        if landmarks:
            mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, f"Prediction: {label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("ASL Real-Time", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

# run formal accuracy test of each label
def run_acc_test():
    cap = cv2.VideoCapture(0)

    accuracies = dict()
    
    # for each letter
    for i, testing_label in enumerate(label_classes):
        fps = 30
        seconds_per_label = 5
        num_correct = 0
        total_frames = fps * seconds_per_label

        # run for total_frames frames
        for frame_num in range(total_frames):
            # bool - is last frame before switching to next label
            is_last_frame = (frame_num + 1 == fps * seconds_per_label)

            ret, frame = cap.read()
            if not ret:
                break
            
            # get keypoints
            keypoints, landmarks = extract_keypoints_from_frame(frame)

            # if a hand detected, use keypoints to predict
            if keypoints:
                x = torch.tensor(keypoints, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    outputs = model(x)
                    _, pred = torch.max(outputs, 1)
                    label = label_classes[pred.item()]
            # otherwise predict "nothing"
            else:
                label = "nothing"

            # if correct prediction
            if label == testing_label:
                num_correct += 1

            # draw landmarks on image if any detected
            if landmarks:
                mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

            # add text showing actual and predicted labels
            cv2.putText(frame, f"Testing: {testing_label}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Prediction: {label}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # add text prepping people for next letter on last frame of current letter
            if is_last_frame:
                cv2.putText(frame, f"Press any key to test the next class. Remember to rotate your hand.", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # if not last class, display next class
                if i + 1 < len(label_classes):
                    cv2.putText(frame, f"Next: {label_classes[i+1]}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # display frame
            cv2.imshow("ASL Real-Time Test", frame)

            # if at the end of a letter
            if is_last_frame:
                # wait until any key is pressed to continue to next letter
                cv2.waitKey(0)
            else:
                # necessary to make it show frames
                cv2.waitKey(1)

        # calculate accuracy of label
        accuracies[testing_label] = num_correct / total_frames
    
    # print per-letter and total accuracies
    print("Accuracies:")
    for letter, acc in accuracies.items(): print(f"{letter}: {acc}")
    total_acc = np.average(accuracies.values())
    print(f"Total: {total_acc}")

    # release window
    cap.release()


# run realtime accuracy test
if args.acc_test:
    run_acc_test()
# run continuous realtime evaluator that stops on q quit key
else:
    run_continuous()

# destory all windows
cv2.destroyAllWindows()