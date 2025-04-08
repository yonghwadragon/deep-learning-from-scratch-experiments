# 6_batch_gradient_descent_visual.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rc

plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 깨짐 방지

# 입력과 정답 정의 (y = 2x 직선)
x = np.array([1.0, 2.0, 3.0])
y_true = np.array([2.0, 4.0, 6.0])

# 초기 가중치와 편향
w = 0.0
b = 0.0

# 학습률 및 반복 횟수
lr = 0.1
epochs = 10

# 직선 기록용
line_history = []

# 손실 및 파라미터 기록용
history = []

# 학습
for epoch in range(epochs):
    y_pred = w * x + b
    loss = np.mean((y_true - y_pred) ** 2)
    dL_dw = np.mean(-2 * x * (y_true - y_pred))
    dL_db = np.mean(-2 * (y_true - y_pred))

    w -= lr * dL_dw
    b -= lr * dL_db

    # 시각화용 직선 저장
    line_x = np.linspace(0, 4, 100)
    line_y = w * line_x + b
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
plt.title("직선이 점들에 가까워지는 과정 (Batch GD)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()