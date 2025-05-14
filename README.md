# ASL_Classification
Final Project for CS231N: Neural Networks for Computer Vision

Made with Python 3.8.18

## 1. Clone the repo
```git clone https://github.com/ElisabethHolm/ASL_Classification.git```

## 2. Install Requirements
```pip install -r requirements.txt```

## 3. Download Dataset
### Get a kaggle API token
Follow these official instructions: https://www.kaggle.com/docs/api#authentication 

tldr: 

Make and download an API token (kaggle.json) on your kaggle account page

Make a ~/.kaggle folder and move the kaggle.json file inside the folder

# YOLO
## Run preprocess_data.py
```python preprocess_data.py```

This downloads and prepares the [ASL alphabet dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet/data) for use with YoloV11

## 4. Train with YOLOv11
Armando either do python or CLI for this, from https://docs.ultralytics.com/tasks/classify/

___________
# MediaPipe
## 1. Enter the mediapipe folder
```cd mediapipe```

## 2. Download base model
Run the following command to download the base model
```wget -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task```

## 3. Download dataset and extract keypoints
```python extract_keypoints.py```

# 4. Train a PyTorch model on keypoints
```python train.py```

# 5. Run real-time inference
```python realtime_test.py```
