from ultralytics import YOLO

model = YOLO("yolo11n-cls.pt")

results = model.train(
    data="alphabet_dataset", 
    epochs=20,
    imgsz=224,             
    batch=32,
    workers=4
)

print("Training complete.")
print(f"Top-1 Accuracy: {results.top1}")
print(f"Top-5 Accuracy: {results.top5}")  