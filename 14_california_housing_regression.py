# 14_california_housing_regression.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. 데이터 로드
housing = fetch_california_housing()
X, y = housing.data, housing.target
# y: 집값 (회귀 문제)
# X: 8개 특성 (예: 인구밀도, 평균 방 갯수 등)

# 2. 데이터 분할
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 전처리 (표준화)
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_val = scaler_X.transform(X_val)

# 타겟도 스케일링해도 되지만, 여기서는 그대로 사용
y_train = y_train.reshape(-1, 1)
y_val = y_val.reshape(-1, 1)

# 텐서 변환
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

# 4. 모델 정의
class Regressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

model = Regressor(X_train.shape[1])  # input_dim=8
criterion = nn.MSELoss()             # 회귀에는 보통 MSE
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. 학습 루프
epochs = 50
train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    # 순전파
    pred_train = model(X_train_tensor)
    loss_train = criterion(pred_train, y_train_tensor)
    
    # 역전파
    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()
    
    train_losses.append(loss_train.item())
    
    # 검증 손실
    model.eval()
    with torch.no_grad():
        pred_val = model(X_val_tensor)
        loss_val = criterion(pred_val, y_val_tensor)
    val_losses.append(loss_val.item())
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {loss_train.item():.4f}, Val Loss: {loss_val.item():.4f}")

# 6. 손실 시각화
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title("California Housing - MLP Regression Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 7. 최종 평가 (RMSE)
with torch.no_grad():
    model.eval()
    y_pred_val = model(X_val_tensor).numpy()
rmse = np.sqrt(np.mean((y_pred_val - y_val)**2))
print(f"\n최종 RMSE on Validation Set: {rmse:.4f}")
