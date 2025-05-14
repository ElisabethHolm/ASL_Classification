# ASL_Classification
Final Project for CS231N: Neural Networks for Computer Vision

Made with Python 3.8.18
# General Set-Up

## 1. Clone the repo
```
git clone https://github.com/ElisabethHolm/ASL_Classification.git
```

## 2. Install Requirements
```
pip install -r requirements.txt
```

## 3. Download Dataset
### Get a kaggle API token
Follow these official instructions: https://www.kaggle.com/docs/api#authentication 

tldr: 

Make and download an API token (kaggle.json) on your kaggle account page

Make a ~/.kaggle folder and move the kaggle.json file inside the folder

_________
# YOLO Classifier
## 1. Run preprocess_data.py
```
python preprocess_data.py
```

This downloads and prepares the [ASL alphabet dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet/data) for use with YoloV11

## 2. Train with YOLOv11
Armando TODO, from https://docs.ultralytics.com/tasks/classify/

___________
# MediaPipe Classifier
## 1. Enter the mediapipe folder
```
cd mediapipe
```
Don't skip this or the cwd gods will unleash their wrath upon you
## 2. Download base model
Run the following command to download the base model
```
wget -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

## 3. Download the asl alphabet dataset
```
python download_dataset.py
```
Note: this may take a second
## 4. Extract keypoints from the dataset (can skip if using existing model)
```
python extract_keypoints.py
```
Note: this takes a bit, maybe go watch a youtube video and come back
# 5. Train a PyTorch model on keypoints (can skip if using existing model)
```
python train.py
```

# 6. Run real-time inference
```
python realtime_test.py
```
If you're on a mac and don't want it to connect to your phone/you want to use the webcam, turn off bluetooth.  
To quit, press the q key.
