# 12_3_xor_activation_boundary_torch.py
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# 한글 폰트 설정
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

# XOR 데이터 정의
x_data = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]], dtype=torch.float32)
y_data = torch.tensor([[0.], [1.], [1.], [0.]], dtype=torch.float32)

# 시각화용 결과 저장
torch_results = {}

# 활성화 함수
activations = {
    'Sigmoid': nn.Sigmoid,
    'Tanh': nn.Tanh,
    'ReLU': nn.ReLU
}

# 학습 파라미터
epochs = 5000
lr = 0.1

for name, act_fn in activations.items():
    class XORModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 4),
                act_fn(),
                nn.Linear(4, 1),
                nn.Sigmoid()
            )
        def forward(self, x):
            return self.net(x)

    model = XORModel()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_list = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(x_data)
        loss = criterion(output, y_data)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())

    with torch.no_grad():
        model.eval()
        pred = model(x_data)
        predicted = (pred > 0.5).float()
        acc = (predicted == y_data).float().mean().item()

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

# 결정 경계 시각화 함수
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

# 손실 시각화
plt.figure(figsize=(10, 5))
for name in torch_results:
    plt.plot(torch_results[name]["loss_list"], label=name)
plt.title("XOR 문제 - 활성화 함수별 손실 감소")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 결과 출력 및 결정 경계
X_np = x_data.numpy()
Y_np = y_data.numpy()
for name in torch_results:
    print(f"\n{name} 결과:")
    print("예측 확률:\n", np.round(torch_results[name]["pred"], 3))
    print("예측 클래스:\n", torch_results[name]["predicted"].flatten())
    print("정확도:", torch_results[name]["accuracy"])
    plot_decision_boundary(torch_results[name]['predict_fn'], f"{name} 결정 경계", X_np, Y_np)
