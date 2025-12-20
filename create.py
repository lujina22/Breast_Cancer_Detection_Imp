import torch
import torch.nn as nn
from torchvision import models

# Load pretrained ResNet50 (ImageNet weights)
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

# Replace the final layer for 4 classes
model.fc = nn.Linear(model.fc.in_features, 3)

# Save the model weights to a .pth file
torch.save(model.state_dict(), "resnet50_mammogram.pth")

print("resnet50_mammogram.pth created successfully!")
