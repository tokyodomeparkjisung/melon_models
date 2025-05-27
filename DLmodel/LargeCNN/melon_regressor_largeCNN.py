import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import pandas as pd


from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision.models.resnet import ResNet18_Weights
from torchvision.models.vgg import VGG16_Weights
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.colors as mcolors


import json
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, data_info, transform=None):
        """
        data_info: 이미지 파일 경로와 레이블 정보가 담긴 딕셔너리
        transform: 이미지에 적용할 전처리
        """
        self.img_labels = data_info
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = list(self.img_labels.keys())[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.img_labels[img_path]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label



class LargeCNN(nn.Module):
    def __init__(self):
        super(LargeCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 16 * 16, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 5)  # Assuming 5 classes
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 256 -> 128
        x = self.pool(F.relu(self.conv2(x)))  # 128 -> 64
        x = self.pool(F.relu(self.conv3(x)))  # 64 -> 32
        x = self.pool(F.relu(self.conv4(x)))  # 32 -> 16
        x = x.view(-1, 256 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x) 
        return x

# Device setup: use MPS on Mac if available, else CUDA, else CPU
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f"Using device: {device}")

# Load JSON data to extract category information for each image
with open('/Users/yun-seungcho/Desktop/고려대학교/멜론/yolov9-main/uniform_data/combined_output.json', 'r') as file:
    data = json.load(file)

# Example dictionary to store image paths and their corresponding categories
image_categories2 = {}

# Assuming 'categories' is the key where category information is stored for each image
# and that each image item in the JSON has a 'category_id' or similar

for anno in data['annotations']:
    for img in data['images']:
        if img['id'] == anno['image_id']:
            image_path = os.path.join('/Users/yun-seungcho/Desktop/고려대학교/멜론/yolov9-main/uniform_data/crops', img['file_name'])

    category_id = anno.get('category_id', 'default_category')  # default if no category present
    image_categories2[image_path] = category_id



transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 데이터셋 생성
dataset = CustomDataset(data_info=image_categories2, transform=transform)

# 데이터셋 분할
total_size = len(dataset)
train_size = int(0.7 * total_size)
val_size = int(0.2 * total_size)
test_size = total_size - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# DataLoader 생성
train_loader256 = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader256 = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader256 = DataLoader(test_dataset, batch_size=1, shuffle=False)


model256 = LargeCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model256.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

def train_model256(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        model.eval()  # Set model to evaluate mode
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
        scheduler.step()
        print(f'Epoch {epoch+1}: Train Loss: {loss.item():.8f}, Validation Loss: {val_loss / len(val_loader):.8f}')
    
    # Save model state
    torch.save(model.state_dict(), 'large_cnn_model.pth')
    print("Model saved as 'large_cnn_model.pth'")



# Start training
train_model256(model256, train_loader256, val_loader256, criterion, optimizer, num_epochs=10)