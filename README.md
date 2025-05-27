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

## 3. Get a kaggle API token
Follow these official instructions: https://www.kaggle.com/docs/api#authentication 

tldr: 

Make and download an API token (kaggle.json) on your kaggle account page

Make a ~/.kaggle folder and move the kaggle.json file inside the folder

_________
# YOLO Classifier
## 1. Download the asl alphabet datasets
```
python preprocess_data.py
```

This downloads and prepares the [ASL alphabet dataset 1](https://www.kaggle.com/datasets/grassknoted/asl-alphabet/data), [ASL alphabet dataset 2](https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset), and the combined version of datasets 1 and 2 for use with YoloV11

The resulting structure is as follows:
```
alphabet_datasets/
├── dataset1/
│   ├── train/
│   ├── val/
│   └── test/
├── dataset2/
│   ├── train/
│   ├── val/
│   └── test/
└── combined/
    ├── train/
    ├── val/
    └── test/
```
Note: this takes a bit

## 2. Train with YOLOv11

```
python train_yolo.py
python evaluate.py
```

From https://docs.ultralytics.com/tasks/classify/

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

## 3. Download the asl alphabet datasets
```
python download_dataset.py
```
This downloads and prepares the [ASL alphabet dataset 1](https://www.kaggle.com/datasets/grassknoted/asl-alphabet/data), [ASL alphabet dataset 2](https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset), and the combined version of datasets 1 and 2 for use with MediaPipe and PyTorch

The resulting structure is as follows:
```
alphabet_datasets/
├── dataset1/
│   ├── A/
│   ├── B/
│   └── ...
├── dataset2/
│   ├── A/
│   ├── B/
│   └── ...
└── combined/
│   ├── A/
│   ├── B/
│   └── ...
```
Note: this takes a bit
## 4. Extract keypoints from the dataset (can skip if using existing model or keypoints)
```
python extract_keypoints.py
```
Note: this takes a while, maybe go watch a youtube video and come back
# 5. Train a PyTorch model on keypoints (can skip if using existing model)
```
python train.py --dataset 1
```
1 = train on dataset 1  
2 = train on dataset 2  
3 = train on combined datasets  

# 6. Run real-time inference
```
python realtime_test.py --training_data 1
```
Again:  
1 = model trained on dataset 1  
2 = model trained on dataset 2  
3 = model trained on combined datasets  

If you're on a mac and don't want it to connect to your phone/you want to use the webcam, turn off bluetooth.  
To quit, press the q key.
