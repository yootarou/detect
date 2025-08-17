import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet


# パラメータ設定
batch_size = 16
epochs = 50
patience = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = 'efficientnet-b0'

# データセットやモデル構築もここに含める
# =====================
# データ前処理（訓練データはデータ拡張）
# =====================
train_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# =====================
# データセット
# =====================
train_dir = "/Users/suenatiyoutarou/Library/CloudStorage/OneDrive-個人用/dataset/train"
val_dir   = "/Users/suenatiyoutarou/Library/CloudStorage/OneDrive-個人用/dataset/valid"

train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
val_dataset   = datasets.ImageFolder(val_dir, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# =====================
# モデル構築（Dropout追加）
# =====================
model = EfficientNet.from_pretrained(model_name)
num_ftrs = model._fc.in_features
model._fc = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(num_ftrs, 2)
)
model = model.to(device)

# =====================
# 損失関数・最適化
# =====================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

# =====================
# EarlyStopping用変数
# =====================
best_val_acc = 0.0
counter = 0

# =====================
# 学習ループ
# =====================
for epoch in range(epochs):
    # ------- 訓練 -------
    model.train()
    running_loss, correct = 0.0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = correct / len(train_dataset)

    # ------- 検証 -------
    model.eval()
    val_loss, val_correct = 0.0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(1)
            val_correct += (preds == labels).sum().item()
    val_loss /= len(val_dataset)
    val_acc = val_correct / len(val_dataset)

    # 学習率スケジューラ更新
    scheduler.step(val_acc)

    print(f"Epoch {epoch+1}/{epochs} "
          f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f} "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    # EarlyStopping判定
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        counter = 0
        torch.save(model.state_dict(), "best_model.pth")  # ベストモデル保存
    else:
        counter += 1
        if counter >= patience:
            print(f"🛑 EarlyStopping: {patience} epochs with no improvement.")
            break
print(f"✅ 学習終了。ベストモデルは best_model.pth に保存されました。")