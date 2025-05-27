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
        x = x.view(-1, 256 * 16 * 16)  # Flatten the layer
        x = F.relu(self.fc1(x))
        x = self.dropout(x) # Apply dropout, especially useful in large networks "to prevent overfitting"
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

# make dataset
dataset = CustomDataset(data_info=image_categories2, transform=transform)

# split dataset
test_size2 = len(dataset)

# make dataloader
test_loader2 = DataLoader(dataset, batch_size=1, shuffle=False)


model256 = LargeCNN().to(device)
model256.load_state_dict(torch.load('melon_regressor_largeCNN.pth', map_location=device))

# 모델을 평가 모드로 설정
model256.eval()

def imshow(img, label, predicted):
    img = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean  # 정규화된 데이터를 원래 값으로 변환
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    plt.title(f'Actual: {label}, Predicted: {predicted}')
    plt.show()

# 성능 평가를 위한 변수 초기화
total = 0
correct = 0

all_preds256 = []
all_labels256 = []

# 테스트 루프
with torch.no_grad():  # 기울기 계산 비활성화
    for images, labels in test_loader2:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model256(images)
        softmax_outputs = F.softmax(outputs, dim=1)
        outputs_percent = softmax_outputs * 100
        outputs_percent = outputs_percent.flatten()
        _, predicted = torch.max(outputs.data, 1)
        all_preds256.append(predicted.item())  # 예측값 저장
        all_labels256.append(labels.item())  # 실제 레이블 저장
        #print(f'Percentage of classes: {outputs_percent[0]:.2f}%, {outputs_percent[1]:.2f}%, {outputs_percent[2]:.2f}%, {outputs_percent[3]:.2f}%, {outputs_percent[4]:.2f}%')
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        #if predicted != labels:
        #    imshow(images[0], labels.item(), predicted.item())
        #imshow(images[0], labels.item(), predicted.item())

# 정확도 계산
accuracy = 100 * correct / total
print(f'Accuracy on test data: {accuracy:.2f}%\n')
print('-----------------------------------------------------------')


# 분류 보고서 출력
print('\nClassification Report\n')
classification_rep = classification_report(all_labels256, all_preds256, output_dict=True)
print(classification_report(all_labels256, all_preds256))

# 데이터프레임 생성
classification_df = pd.DataFrame(classification_rep).transpose()

# 정확도를 데이터프레임으로 추가
accuracy_df = pd.DataFrame({'Accuracy': [accuracy]})

# 엑셀로 저장
with pd.ExcelWriter('LargeCNN_model_evaluation.xlsx') as writer:
    accuracy_df.to_excel(writer, sheet_name='Accuracy', index=False)
    classification_df.to_excel(writer, sheet_name='Classification Report')

print("Evaluation results saved to 'LargeCNN_model_evaluation.xlsx'")
