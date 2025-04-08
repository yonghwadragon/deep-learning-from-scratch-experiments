# 15_overfitting_control_methods.py

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# --------------------------
# 1. 과적합 유도용 가짜 데이터 생성
# --------------------------
np.random.seed(0)
X_all = np.random.uniform(-1, 1, size=(20, 2))
y_all = X_all[:,0]**2 + X_all[:,1]*0.5 + np.random.normal(0, 0.05, size=(20,))

indices = np.arange(len(X_all))
np.random.shuffle(indices)
train_indices = indices[:15]
test_indices  = indices[15:]
X_train, y_train = X_all[train_indices], y_all[train_indices]
X_test,  y_test  = X_all[test_indices],  y_all[test_indices]

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor  = torch.tensor(X_test,  dtype=torch.float32)
y_test_tensor  = torch.tensor(y_test,  dtype=torch.float32).unsqueeze(1)

# --------------------------
# 2. 기법별 모델 생성 함수
# --------------------------
def build_model(use_dropout=False, use_batchnorm=False, l2_reg=0.0):
    layers = []
    input_dim = 2
    hidden_dim = 64
    
    layers.append(nn.Linear(input_dim, hidden_dim))
    if use_batchnorm:
        layers.append(nn.BatchNorm1d(hidden_dim))
    layers.append(nn.ReLU())
    if use_dropout:
        layers.append(nn.Dropout(p=0.5))
    layers.append(nn.Linear(hidden_dim, 1))
    
    model = nn.Sequential(*layers)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=l2_reg)
    return model, optimizer

# --------------------------
# 3. 학습 함수
# --------------------------
def train_model(model, optimizer, epochs=300):
    criterion = nn.MSELoss()
    train_losses = []
    test_losses = []
    for _ in range(epochs):
        # Train
        model.train()
        pred_train = model(X_train_tensor)
        loss_train = criterion(pred_train, y_train_tensor)
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        train_losses.append(loss_train.item())
        # Test
        model.eval()
        with torch.no_grad():
            pred_test = model(X_test_tensor)
            loss_test = criterion(pred_test, y_test_tensor)
        test_losses.append(loss_test.item())
    return train_losses, test_losses

# --------------------------
# 4. 실험 설정
# --------------------------
configs = [
    {"name": "No Regularization", "dropout": False, "bn": False, "l2": 0.0},
    {"name": "Dropout",           "dropout": True,  "bn": False, "l2": 0.0},
    {"name": "BatchNorm",         "dropout": False, "bn": True,  "l2": 0.0},
    {"name": "L2=1e-3",           "dropout": False, "bn": False, "l2": 1e-3},
]

all_results = {}  # 예: { "No Regularization": {"train": [...], "test": [...]} }

# --------------------------
# 5. 각 기법 학습
# --------------------------
for cfg in configs:
    model, optimizer = build_model(use_dropout=cfg["dropout"], 
                                   use_batchnorm=cfg["bn"],
                                   l2_reg=cfg["l2"])
    train_losses, test_losses = train_model(model, optimizer, epochs=300)
    all_results[cfg["name"]] = {
        "train": train_losses,
        "test":  test_losses
    }

# ============================
# 6. 그래프 1: 4개 설정의 Train만 (4줄)
# ============================
plt.figure(figsize=(8, 5))
for name, data in all_results.items():
    plt.plot(data["train"], label=f"{name} (Train)")
plt.title("Graph 1: Train Loss Only (4 lines)")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)
plt.show()

# =========================================
# 7. 그래프 2~5: 각 설정별로 Train vs Test
# =========================================
for i, cfg in enumerate(configs):
    name = cfg["name"]
    plt.figure(figsize=(8, 5))
    plt.plot(all_results[name]["train"], label=f"{name} Train")
    plt.plot(all_results[name]["test"],  label=f"{name} Test", linestyle='--')
    plt.title(f"Graph {i+2}: {name} (Train vs Test)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

# =========================================
# 8. 그래프 6: 모든 설정 + (Train & Test) ⇒ 8줄
# =========================================
plt.figure(figsize=(10, 6))
for name, data in all_results.items():
    # train
    plt.plot(data["train"], label=f"{name} (Train)")
    # test
    plt.plot(data["test"], linestyle='--', label=f"{name} (Test)")
plt.title("Graph 6: All Methods (Train & Test)")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)
plt.show()

# ============================
# 9. 최종 비교
# ============================
print("\n===== 최종 손실 비교 (Epoch 300) =====")
for cfg in configs:
    name = cfg["name"]
    train_final = all_results[name]["train"][-1]
    test_final  = all_results[name]["test"][-1]
    print(f"{name}: Train={train_final:.6f}, Test={test_final:.6f}")
    
"""
- No Regularization: Overfitting 발생 → Train Loss 훨씬 낮아지고, Test Loss 크면 오버피팅
- Dropout: 과적합 완화
- BatchNorm: 입력 분포 정규화 → 학습 안정화, 과적합 완화
- L2=1e-3: Weight Decay로 과적합 완화
"""