from ultralytics import YOLO
from pathlib import Path
import random

# Load trained model (replace with your actual path if needed)
model = YOLO("runs/classify/train/weights/best.pt")

# Validate on val set (MNIST-style folder structure is remembered)
metrics = model.val()
print(f"Top-1 Accuracy: {metrics.top1}")
print(f"Top-5 Accuracy: {metrics.top5}")

# predict a few samples, testing
val_dir = Path("alphabet_dataset/val")
class_folders = sorted(val_dir.iterdir())
num_samples = 5

print("\n🔍 Sample Predictions:")
for _ in range(num_samples):
    class_folder = random.choice(class_folders)
    true_class = class_folder.name
    img_path = random.choice(list(class_folder.glob("*.jpg")))

    result = model(str(img_path))[0]
    predicted_class = result.names[result.probs.top1]

    print(f"🖼️ Image: {img_path.name}")
    print(f"🔹 True: {true_class} | 🔸 Predicted: {predicted_class} | Confidence: {result.probs.top1conf:.2%}\n")