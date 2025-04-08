# 06_2_batch_gradient_descent_visual.py
"""일부로 안되는 예시
출력층에 시그모이드를 사용하면 회귀 문제에서 큰 값을 예측할 수 없음.
적합하지 않다는 점을 확인.
추가로, 만약 회귀 문제에서 시그모이드 출력이 필요하다면,
출력층에 선형 활성화 함수를 사용하거나,
시그모이드 출력 후 스케일링하는 방법을 사용."""
"""어떤 수학 함수든 다음 조건만 만족하면
직접 만들어서 사용할 수 있습니다.

✅ 좋은 활성화 함수의 조건:
비선형성: 직선이 아닌 곡선 형태

미분 가능성: 역전파(backpropagation)에 사용되므로

출력 안정성: 너무 크거나 작지 않게 출력 조절"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rc

plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 깨짐 방지

# 데이터 정의 (y = 2x 직선)
x = np.array([1.0, 2.0, 3.0])
y_true = np.array([2.0, 4.0, 6.0])

# 활성화 함수: Sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 초기 가중치와 편향
w = 0.0
b = 0.0

# 학습률 및 반복 횟수
lr = 0.1
epochs = 10

# 직선 기록용 (여기서는 sigmoid 적용된 선)
line_history = []

# 손실 및 파라미터 기록용
history = []

# 학습
for epoch in range(epochs):
    # 시그모이드 활성화 적용: y_pred = sigmoid(w*x + b)
    y_pred = sigmoid(w * x + b)
    loss = np.mean((y_true - y_pred) ** 2)
    dL_dw = np.mean(-2 * x * (y_true - y_pred) * y_pred * (1 - y_pred))  # 체인룰 적용
    dL_db = np.mean(-2 * (y_true - y_pred) * y_pred * (1 - y_pred))

    w -= lr * dL_dw
    b -= lr * dL_db

    # 시각화용 직선 저장: x값 범위는 0~4
    line_x = np.linspace(0, 4, 100)
    line_y = sigmoid(w * line_x + b)
    line_history.append((line_x, line_y, f"Epoch {epoch+1}"))

    # 터미널 출력용 기록
    history.append({
        'epoch': epoch + 1,
        'w': round(w, 4),
        'b': round(b, 4),
        'loss': round(loss, 6)
    })

# 학습 결과 테이블 출력
df = pd.DataFrame(history)
print("[학습 이력]")
print(df[['epoch', 'w', 'b', 'loss']])

# 시각화 출력
plt.figure(figsize=(8, 6))
plt.scatter(x, y_true, color='black', label='True Data (y = 2x)')
for i, (lx, ly, label) in enumerate(line_history):
    if i in [0, 2, 4, 9]:
        plt.plot(lx, ly, label=label)
plt.title("시그모이드 적용 시 예측 직선 (출력 0~1 범위)")
plt.xlabel("x")
plt.ylabel("y_pred = sigmoid(w*x+b)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()