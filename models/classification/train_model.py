import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# ---------------- CONFIG ----------------
DATA_DIR = "data/raw"
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 25
LR = 1e-4

# ---------------- CUSTOM DATASET ----------------
class MRIDataset(Dataset):
    def __init__(self, images, labels, augment=False):
        self.images = images
        self.labels = labels
        self.augment = augment
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        self.basic = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.augment:
            img = self.transform(img)
        else:
            img = self.basic(img)
        return img, self.labels[idx]

# ---------------- LOAD DATA ----------------
images, labels = [], []

for label, folder in enumerate(["no", "yes"]):
    for img_name in os.listdir(os.path.join(DATA_DIR, folder)):
        img_path = os.path.join(DATA_DIR, folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        images.append(img)
        labels.append(label)

images = np.array(images)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, stratify=labels, random_state=42
)

train_ds = MRIDataset(X_train, y_train, augment=True)
test_ds = MRIDataset(X_test, y_test, augment=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

# ---------------- MODEL ----------------
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

# Freeze most layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze last block
for param in model.features[-1].parameters():
    param.requires_grad = True

model.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.classifier[1].in_features, 2)
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ---------------- LOSS & OPTIMIZER ----------------
class_weights = torch.tensor([1.5, 1.0]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

# ---------------- TRAINING ----------------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, lbls)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss/len(train_loader):.4f}")

# ---------------- EVALUATION ----------------
model.eval()
correct, total = 0, 0

with torch.no_grad():
    for imgs, lbls in test_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)
        total += lbls.size(0)
        correct += (preds == lbls).sum().item()

accuracy = 100 * correct / total
print(f"🔥 FINAL TEST ACCURACY: {accuracy:.2f}%")

torch.save(model.state_dict(), "models/classification/brain_tumor_model.pth")
print("✅ EfficientNet model saved!")
