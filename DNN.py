# ================================
# Pretrained DNN Model (ResNet50)
# Evaluation on Full Dataset
# ================================

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm

# --------------------------------
# 1. Load Trained Model
# --------------------------------
def load_dnn_model(model_path):
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 3)  # Normal, Mass, Micro
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
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
    return img.unsqueeze(0)

# --------------------------------
# 3. Predict Single Image
# --------------------------------
def predict_single(model, image_path):
    with torch.no_grad():
        input_tensor = preprocess_image(image_path)
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).item()
    return pred

# --------------------------------
# 4. Evaluate on Dataset
# --------------------------------
def evaluate_dataset(csv_file, model_path="resnet50_mammogram.pth"):
    # Load model
    model = load_dnn_model(model_path)

    # Load CSV
    df = pd.read_csv(csv_file)

    # 🔴 CHANGE THIS if your column name is different
    image_paths = df["PATH"].tolist()
    labels_text = df["TYPE"].tolist()

    class_map = {
        "NORMAL": 0,
        "MASS": 1,
        "MICRO_CALCIFICATION": 2
    }

    correct = 0
    total = len(image_paths)

    for img_path, label_text in tqdm(zip(image_paths, labels_text), total=total):
        true_label = class_map[label_text]
        pred_label = predict_single(model, img_path)

        if pred_label == true_label:
            correct += 1

    accuracy = (correct / total) * 100
    print("\n==============================")
    print(f"Total images : {total}")
    print(f"Correct      : {correct}")
    print(f"Accuracy     : {accuracy:.2f}%")
    print("==============================")

# --------------------------------
# 5. Main
# --------------------------------
if __name__ == "__main__":
    csv_file = "train_dataset.csv"   # or test_dataset.csv
    model_path = "resnet50_mammogram.pth"

    evaluate_dataset(csv_file, model_path)
