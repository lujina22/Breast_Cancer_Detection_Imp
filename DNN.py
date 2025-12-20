# ================================
# Pretrained DNN Model (ResNet50)
# Updated for 3 Classes: Normal, Mass, Micro
# ================================

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# --------------------------------
# 1. Load Pretrained Model
# --------------------------------
def load_dnn_model(model_path):
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 3)  # 3 classes
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# --------------------------------
# 2. Image Preprocessing
# --------------------------------
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    img = Image.open(image_path).convert("L")
    img = transform(img)
    img = img.unsqueeze(0)  # Add batch dimension
    return img

# --------------------------------
# 3. Run DNN Prediction
# --------------------------------
def predict(image_path, model_path="resnet50_mammogram.pth"):
    classes = ["Normal", "Mass", "Micro"]  # 3 classes

    model = load_dnn_model(model_path)
    input_tensor = preprocess_image(image_path)

    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1)

    return classes[prediction.item()]

# --------------------------------
# 4. Main (Example Usage)
# --------------------------------
if __name__ == "__main__":
    image_path = "Dataset/all-mias/mdb256.pgm"  # Replace with your test image
    model_path = "resnet50_mammogram.pth"      # Ensure this is trained with 3 classes

    result = predict(image_path, model_path)
    print("DNN Prediction:", result)
