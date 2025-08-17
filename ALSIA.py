from PIL import Image
import torch
import torch.nn as nn  # ← これが必要
from torchvision import transforms
from efficientnet_pytorch import EfficientNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# モデル構築（学習時と同じ構造）
# =====================
model = EfficientNet.from_pretrained("efficientnet-b0")
num_ftrs = model._fc.in_features
model._fc = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(num_ftrs, 2)  # 2クラス分類
)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()
model.to(device)

# =====================
# 前処理
# =====================
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =====================
# 推論したい画像
# =====================
image_path = "/Users/suenatiyoutarou/Library/CloudStorage/OneDrive-個人用/dataset/valid/healthy/スクリーンショット 2025-08-16 17.49.09.png"
img = Image.open(image_path).convert("RGB")
img = transform(img).unsqueeze(0).to(device)

# =====================
# 推論
# =====================
with torch.no_grad():
    output = model(img)
    probs = torch.softmax(output, dim=1)  # 2クラスなので softmax
    predicted_class = torch.argmax(probs, dim=1).item()

class_names = ["diseased", "healthy"]
print(f"予測: {class_names[predicted_class]}, 確率: {probs[0][predicted_class]:.4f}")