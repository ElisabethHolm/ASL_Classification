import cv2
import torch
import torch.nn as nn
import mediapipe as mp
import numpy as np
import json

with open("label_classes.json", "r") as f:
    label_classes = json.load(f)

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ASLClassifier()
model.load_state_dict(torch.load("mp_alphabet_classifier.pth", map_location=device))
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
cv2.destroyAllWindows()