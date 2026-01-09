import os
import cv2
import numpy as np

DATA_DIR = "data/raw"
IMG_SIZE = 224

data = []
labels = []

for label, folder in enumerate(["no", "yes"]):
    folder_path = os.path.join(DATA_DIR, folder)
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0   # normalize

        data.append(img)
        labels.append(label)

data = np.array(data)
labels = np.array(labels)

print("Data shape:", data.shape)
print("Labels shape:", labels.shape)
print("Tumor images:", np.sum(labels == 1))
print("No tumor images:", np.sum(labels == 0))
