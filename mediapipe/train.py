import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import json

test_size = 0.2  # 80% train, remainder split between val (10%) and test (10%)
batch_size = 32
learning_rate = 1e-3
num_epochs = 25

class HandKeypointDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

with open("keypoints_dataset.json") as f:
    raw = json.load(f)

X = np.array(raw["data"])
y = np.array(raw["labels"])

le = LabelEncoder()
y_encoded = le.fit_transform(y)
with open("label_classes.json", "w") as f:
    json.dump(le.classes_.tolist(), f)

# Split: 80% train, 10% val, 10% test
# training (80%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y_encoded, test_size=test_size, random_state=42)
# val and test (10%, 10%)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  # 0.5 * 0.2 = 0.1 each

train_loader = DataLoader(HandKeypointDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(HandKeypointDataset(X_val, y_val), batch_size=32)
test_loader = DataLoader(HandKeypointDataset(X_test, y_test), batch_size=32)

class ASLClassifier(nn.Module):
    def __init__(self, input_dim=63, num_classes=26):
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

def compute_accuracy(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ASLClassifier(input_dim=X.shape[1], num_classes=len(le.classes_)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(inputs), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_acc = compute_accuracy(model, train_loader, device)
    val_acc = compute_accuracy(model, val_loader, device)
    print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

# Final test accuracy
test_acc = compute_accuracy(model, test_loader, device)
print(f"\nFinal Test Accuracy: {test_acc:.2f}%")

torch.save(model.state_dict(), "mp_alphabet_classifier.pth")