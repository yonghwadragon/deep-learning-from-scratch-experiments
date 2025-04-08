# 12_3_xor_activation_boundary_torch.py
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# 한글 폰트 설정 (NanumGothic 없으면 기본 폰트 사용)
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

# ▣ 랜덤 시드 고정
torch.manual_seed(0)
np.random.seed(0)

# XOR 데이터 정의
x_data = torch.tensor([[0., 0.],
                       [0., 1.],
                       [1., 0.],
                       [1., 1.]], dtype=torch.float32)
y_data = torch.tensor([[0.],
                       [1.],
                       [1.],
                       [0.]], dtype=torch.float32)

# 시각화용 결과 저장
torch_results = {}

# 활성화 함수 사전
activations = {
    'Sigmoid': nn.Sigmoid,
    'Tanh': nn.Tanh,
    'ReLU': nn.ReLU
}

# 학습 파라미터
epochs = 10000
lr = 0.01  # 기존 0.1 → 0.01로 변경(약간 더 안정적으로)
hidden_dim = 8  # 은닉층 노드 수

for name, act_fn in activations.items():
    # 2개 은닉층 추가
    class XORModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, hidden_dim),   # 1번 은닉층
                act_fn(),
                nn.Linear(hidden_dim, hidden_dim),  # 2번 은닉층
                act_fn(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()  # 출력층
            )
        def forward(self, x):
            return self.net(x)

    model = XORModel()
    criterion = nn.BCELoss()                     # BCE 손실
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_list = []

    # 학습 루프
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(x_data)
        loss = criterion(output, y_data)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())

    # 평가 단계
    with torch.no_grad():
        model.eval()
        pred = model(x_data)
        predicted = (pred > 0.5).float()
        acc = (predicted == y_data).float().mean().item()

        # 결정경계용 함수
        def predict_fn(x):
            x_tensor = torch.tensor(x, dtype=torch.float32)
            out = model(x_tensor).detach().numpy()
            return (out > 0.5).astype(int)

    torch_results[name] = {
        "loss_list": loss_list,
        "pred": pred.numpy(),
        "predicted": predicted.numpy(),
        "accuracy": acc,
        "predict_fn": predict_fn
    }

# =========================
# 결정 경계 시각화 함수
# =========================
def plot_decision_boundary(model_fn, title, X, Y):
    h = 0.01
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model_fn(grid)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.6)
    plt.scatter(X[:, 0], X[:, 1], c=Y.flatten(), cmap=plt.cm.RdBu, edgecolors='k')
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True)
    plt.show()

# 손실 그래프
plt.figure(figsize=(10, 5))
for name in torch_results:
    plt.plot(torch_results[name]["loss_list"], label=name)
plt.title("XOR 문제 - (은닉 2개) + Adam + BCE Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 결과 출력 & 결정 경계
X_np = x_data.numpy()
Y_np = y_data.numpy()
for name in torch_results:
    print(f"\n[{name}] 결과:")
    print("최종 예측 확률:\n", np.round(torch_results[name]["pred"], 3))
    print("예측 클래스:\n", torch_results[name]["predicted"].flatten())
    print("정확도:", torch_results[name]["accuracy"])
    plot_decision_boundary(torch_results[name]['predict_fn'],
                          f"{name} 결정 경계", X_np, Y_np)