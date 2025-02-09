import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
import sys

# ========== Check Environment (Google Colab or Local) ==========
if 'google.colab' in sys.modules:
    from google.colab import drive
    drive.mount('/content/drive')
    base_path = "/content/drive/MyDrive/dataset/"
else:
    base_path = "./dataset/"  # Modify for local execution

# ========== Dataset Class ==========
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.classes = sorted(os.listdir(root_dir))

        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for image_name in os.listdir(class_dir):
                    if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(class_dir, image_name))
                        self.labels.append(label)
        print(f"Found {len(self.image_paths)} images in {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# ========== Transformations ==========
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ========== Modified ResNet Model ==========
class ModifiedResNet(nn.Module):
    def __init__(self, num_classes):
        super(ModifiedResNet, self).__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)  # Fixes warning
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)

# ========== Training Function ==========
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Store history
    history = {
        'train_acc': [], 'train_f1': [], 'val_acc': [], 'val_f1': [],
        'train_precision': [], 'val_precision': [],
        'train_recall': [], 'val_recall': [],
        'train_roc': [], 'val_roc': []
    }

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        all_train_labels = []
        all_train_preds = []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            all_train_labels.extend(labels.cpu().numpy())
            all_train_preds.extend(preds.cpu().numpy())

        train_acc = accuracy_score(all_train_labels, all_train_preds)
        train_f1 = f1_score(all_train_labels, all_train_preds, average='macro')
        train_recall = recall_score(all_train_labels, all_train_preds, average='macro')
        train_precision = precision_score(all_train_labels, all_train_preds, average='macro')
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['train_recall'].append(train_recall)
        history['train_precision'].append(train_precision)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}")

        # Validation phase
        model.eval()
        all_val_labels = []
        all_val_preds = []
        all_val_probs = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                probs = nn.functional.softmax(outputs, dim=1)

                all_val_labels.extend(labels.cpu().numpy())
                all_val_preds.extend(probs.argmax(dim=1).cpu().numpy())
                all_val_probs.extend(probs.cpu().numpy())

        val_acc = accuracy_score(all_val_labels, all_val_preds)
        val_f1 = f1_score(all_val_labels, all_val_preds, average='macro')
        val_recall = recall_score(all_val_labels, all_val_preds, average='macro')
        val_precision = precision_score(all_val_labels, all_val_preds, average='macro')
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['val_recall'].append(val_recall)
        history['val_precision'].append(val_precision)

        try:
            fpr, tpr, _ = roc_curve(
                label_binarize(all_val_labels, classes=list(range(len(set(all_val_labels))))).ravel(),
                np.array(all_val_probs).ravel()
            )
            history['val_roc'].append((fpr, tpr))
        except ValueError:
            print("Skipping ROC curve calculation due to single class in epoch.")

        print(f"Epoch {epoch + 1}/{num_epochs}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

    return model, history

# ========== Plot Functions ==========
def plot_metrics(history, metric_name, dataset_type):
    plt.figure(figsize=(10, 6))
    plt.plot(history[f'{dataset_type}_{metric_name}'], label=f'{dataset_type.capitalize()} {metric_name.capitalize()}')
    plt.title(f'{metric_name.capitalize()} Over Epochs ({dataset_type.capitalize()})')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name.capitalize())
    plt.legend()
    plt.grid()
    plt.show()

def plot_roc_curves(history, dataset_type='train'):
    roc_data = history[f'{dataset_type}_roc']
    plt.figure(figsize=(10, 8))
    for epoch, (fpr, tpr) in enumerate(roc_data):
        plt.plot(fpr, tpr, label=f'Epoch {epoch + 1}')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.title(f'ROC Curves for {dataset_type.capitalize()} Dataset')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()

# ========== Define Datasets ==========
datasets = {
    "FERPlus": {
        "train": os.path.join(base_path, "FERPlus/train"),
        "val": os.path.join(base_path, "FERPlus/validation")
    } 
}

for dataset_name, paths in datasets.items():
    print(f"Processing dataset: {dataset_name}")

    # Load datasets
    train_dataset = ImageDataset(paths['train'], transform)
    val_dataset = ImageDataset(paths['val'], transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Define model, loss, and optimizer
    num_classes = len(train_dataset.classes)
    model = ModifiedResNet(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    model, history = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)

    # Plot metrics
    for metric in ['acc', 'f1', 'recall', 'precision']:
        plot_metrics(history, metric, dataset_type='train')
        plot_metrics(history, metric, dataset_type='val')

    # Plot ROC curves
    plot_roc_curves(history, dataset_type='train')
    plot_roc_curves(history, dataset_type='val')
