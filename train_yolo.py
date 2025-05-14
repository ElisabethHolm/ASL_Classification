from ultralytics import YOLO
#Classify models are pretrained on the ImageNet dataset.

# Load a pretrained model (for classification)
model = YOLO("yolo11n-cls.pt")

# Train on the preprocessed dataset
results = model.train(
    data="alphabet_dataset", 
    epochs=20,
    imgsz=224,             
    batch=32,
    workers=4
)

# Optional: Print results
print("Training complete.")
print(f"Top-1 Accuracy: {results.top1}")
print(f"Top-5 Accuracy: {results.top5}")  