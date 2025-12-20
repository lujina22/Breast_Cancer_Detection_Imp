import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.model_selection import train_test_split

# ---------------------------
# 1. Custom Dataset Class
# ---------------------------
class MammogramDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("L")  # grayscale
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

# ---------------------------
# 2. Load Dataset from CSV
# ---------------------------
def load_dataset_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    
    # Map class names to integers
    class_map = {
        "NORMAL": 0,
        "MASS": 1,
        "MICRO_CALCIFICATION": 2
    }
    
    image_paths = df["PATH"].tolist()   # replace 'PATH' with your actual image path column name
    labels = [class_map[t] for t in df["TYPE"]]  # 'TYPE' column has the class name
    
    return image_paths, labels, list(class_map.keys())

# ---------------------------
# 3. Data Transforms
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---------------------------
# 4. Training Setup
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
csv_file = "train_dataset.csv"  # your CSV file path

image_paths, labels, class_names = load_dataset_from_csv(csv_file)

# Split train/test
train_paths, test_paths, train_labels, test_labels = train_test_split(
    image_paths, labels, test_size=0.2, stratify=labels, random_state=42
)

train_dataset = MammogramDataset(train_paths, train_labels, transform)
test_dataset = MammogramDataset(test_paths, test_labels, transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# ---------------------------
# 5. Load Pretrained ResNet50
# ---------------------------
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 3)  # 3 classes
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ---------------------------
# 6. Training Loop
# ---------------------------
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels_batch in train_loader:
        images = images.to(device)
        labels_batch = labels_batch.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels_batch.size(0)
        correct += (predicted == labels_batch).sum().item()

    acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss/len(train_loader):.4f} Acc: {acc:.2f}%")

# ---------------------------
# 7. Save Trained Model
# ---------------------------
torch.save(model.state_dict(), "resnet50_mammogram.pth")
print("Model saved as resnet50_mammogram.pth")
