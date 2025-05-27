import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
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



def initialize_resnet(num_classes):
    # ResNet-18 모델 불러오기 (ImageNet으로 사전 훈련된 가중치 사용)
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    
    # 마지막 완전 연결층 변경하기
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)  # 예를 들어, 클래스가 5개라면 num_classes = 5

    return model

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


# 데이터 전처리 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 입력 크기를 224x224로 조정
    transforms.ToTensor(),  # 이미지를 PyTorch 텐서로 변환
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 정규화
])

# 데이터셋 생성
dataset = CustomDataset(data_info=image_categories2, transform=transform)

# 데이터셋 분할
train_size = int(0.7 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# DataLoader 생성
train_loader224 = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader224 = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader224 = DataLoader(test_dataset, batch_size=1, shuffle=False)


# 모델 초기화
num_classes = 5  # 예시: 클래스가 5개인 경우
res_model = initialize_resnet(num_classes)
res_model = res_model.to(device)

# 손실 함수와 최적화기 설정
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(res_model.parameters(), lr=0.001)

def train_res_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
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
                
        print(f'Epoch {epoch+1}: Train Loss: {loss.item():.8f}, Validation Loss: {val_loss / len(val_loader):.8f}')
    
    # Save model state
    torch.save(model.state_dict(), 'melon_regressor_resnet18.pth')
    print("Model saved as 'melon_regressor_resnet18.pth'")



# Start training
train_res_model(res_model, train_loader224, val_loader224, criterion, optimizer, num_epochs=10)