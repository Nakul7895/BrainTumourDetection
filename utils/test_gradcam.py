import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import cv2
import numpy as np
from torchvision import models, transforms
from utils.gradcam import GradCAM


# ---------------- LOAD MODEL ----------------
model = models.efficientnet_b0(
    weights=models.EfficientNet_B0_Weights.DEFAULT
)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)

model.load_state_dict(
    torch.load("models/classification/brain_tumor_model.pth", map_location="cpu")
)
model.eval()

# ---------------- TARGET LAYER ----------------
target_layer = model.features[-1]

gradcam = GradCAM(model, target_layer)

# ---------------- LOAD TEST IMAGE ----------------
img_path = os.path.join("data/raw/yes", os.listdir("data/raw/yes")[0])
img = cv2.imread(img_path)
img = cv2.resize(img, (224, 224))
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

input_tensor = transform(img_rgb).unsqueeze(0)

# ---------------- PREDICTION ----------------
output = model(input_tensor)
class_idx = torch.argmax(output).item()

# ---------------- GRAD-CAM ----------------
heatmap = gradcam.generate(input_tensor, class_idx)

heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
overlay = heatmap * 0.4 + img

# ---------------- DISPLAY ----------------
cv2.imshow("Grad-CAM Result", overlay.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
