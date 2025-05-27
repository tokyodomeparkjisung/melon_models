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

# Custom dataset
class CustomDataset(Dataset):
    def __init__(self, data_info, transform=None):
        self.img_labels = list(data_info.items())  # list of (path, label)
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path, label = self.img_labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label



class SimpleCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 18 * 18, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 18 * 18)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



# Load JSON data to extract category information for each image
with open('/Users/yun-seungcho/Desktop/고려대학교/멜론/yolov9-main/uniform_data/combined_output.json', 'r') as file:
    data = json.load(file)

# Example dictionary to store image paths and their corresponding categories
image_categories = {}

# Assuming 'categories' is the key where category information is stored for each image
# and that each image item in the JSON has a 'category_id' or similar

for anno in data['annotations']:
    for img in data['images']:
        if img['id'] == anno['image_id']:
            image_path = os.path.join('/Users/yun-seungcho/Desktop/고려대학교/멜론/yolov9-main/uniform_data/crops', img['file_name'])

    category_id = anno.get('category_id', 'default_category')  # default if no category present
    image_categories[image_path] = category_id

# Load JSON data to extract category information for each image
with open('/Users/yun-seungcho/Desktop/고려대학교/멜론/yolov9-main/additional_data/crop/results.json', 'r') as file:
    data = json.load(file)

# Example dictionary to store image paths and their corresponding categories
image_categories2 = {}

# Assuming 'categories' is the key where category information is stored for each image
# and that each image item in the JSON has a 'category_id' or similar

for anno in data['annotations']:
    for img in data['images']:
        if img['id'] == anno['image_id']:
            image_path = os.path.join('/Users/yun-seungcho/Desktop/고려대학교/멜론/yolov9-main/additional_data/crop/crops/melon/images', img['file_name'])

    category_id = anno.get('category_id', 'default_category')  # default if no category present
    image_categories2[image_path] = category_id


transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# make dataset
dataset = CustomDataset(data_info=image_categories, transform=transform)

# split dataset
total_size = len(dataset)
train_size = int(0.7 * total_size)
val_size = int(0.2 * total_size)
test_size = total_size - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# make dataloader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model = SimpleCNN(num_classes=len(set(image_categories.values())))
model.load_state_dict(torch.load('melon_regressor_simpleCNN.pth'))


# 모델을 평가 모드로 설정
model.eval()

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

all_preds = []
all_labels = []

# 테스트 루프
with torch.no_grad():  # 기울기 계산 비활성화
    for images, labels in test_loader:
        outputs = model(images)
        softmax_outputs = F.softmax(outputs, dim=1)
        #softmax_outputs = softmax_outputs.flatten()
        outputs_percent = softmax_outputs * 100
        outputs_percent = outputs_percent.numpy().flatten()
        _, predicted = torch.max(outputs.data, 1)
        all_preds.append(predicted.numpy().item(0))  # 예측값 저장
        all_labels.append(labels.numpy().item(0))  # 실제 레이블 저장
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
classification_rep = classification_report(all_labels, all_preds, output_dict=True)
print(classification_report(all_labels, all_preds))

# 데이터프레임 생성
classification_df = pd.DataFrame(classification_rep).transpose()

# 정확도를 데이터프레임으로 추가
accuracy_df = pd.DataFrame({'Accuracy': [accuracy]})

# 엑셀로 저장
with pd.ExcelWriter('SimpleCNN_model_evaluation.xlsx') as writer:
    accuracy_df.to_excel(writer, sheet_name='Accuracy', index=False)
    classification_df.to_excel(writer, sheet_name='Classification Report')

print("Evaluation results saved to 'SimpleCNN_model_evaluation.xlsx'")