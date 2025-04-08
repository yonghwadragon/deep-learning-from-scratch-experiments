# 2_sin_predictor_pytorch.py
import numpy as np                      # 수치 계산을 위한 numpy
import torch                            # 파이토치 불러오기
import torch.nn as nn                   # 신경망 모듈
import matplotlib.pyplot as plt         # 그래프 시각화

x = np.linspace(0, 2 * np.pi, 100)      # x 데이터 (0~2π)
y = np.sin(x)                           # y는 sin(x)

x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(1)  # (100, 1) 텐서
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # (100, 1) 텐서

class SinNet(nn.Module):                         # 신경망 클래스 정의
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(                # 순차적 모델 정의
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

model = SinNet()                                 # 모델 인스턴스 생성
criterion = nn.MSELoss()                         # 손실함수: MSE
optimizer = torch.optim.Adam(model.parameters()) # 옵티마이저: Adam

for epoch in range(1000):                        # 1000번 학습 반복
    y_pred = model(x_tensor)                     # 예측값 계산
    loss = criterion(y_pred, y_tensor)           # 손실 계산
    optimizer.zero_grad()                        # 기울기 초기화
    loss.backward()                              # 역전파
    optimizer.step()                             # 가중치 업데이트

y_output = model(x_tensor).detach().numpy()      # 예측 결과 numpy 변환

plt.plot(x, y, label='Real sin(x)')              # 실제 sin(x)
plt.plot(x, y_output, label='Predicted')         # 예측값
plt.legend()
plt.title("sin(x) Prediction with PyTorch")
plt.show()
