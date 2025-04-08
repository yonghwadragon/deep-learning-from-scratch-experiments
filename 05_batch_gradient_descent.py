# 5_batch_gradient_descent.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 다중 입력 샘플 데이터 정의 (선형 관계)
x = np.array([1.0, 2.0, 3.0])        # 입력값 3개
y_true = np.array([2.0, 4.0, 6.0])   # 정답 (정비례)

# 초기 가중치와 편향
w = 0.0
b = 0.0

# 학습률
lr = 0.1
print(f"학습률 = {lr}\n")

# 저장할 이력
history = []

# 에포크 수
epochs = 10

for epoch in range(epochs):
    # 순전파
    y_pred = w * x + b

    # 손실 (MSE)
    loss = np.mean((y_true - y_pred) ** 2)

    # 미분 (평균 기울기)
    dL_dw = np.mean(-2 * x * (y_true - y_pred))
    dL_db = np.mean(-2 * (y_true - y_pred))

    # 가중치 업데이트
    w_old, b_old = w, b
    w -= lr * dL_dw
    b -= lr * dL_db

    # 출력
    print(f"[Epoch {epoch + 1}]")
    print(f"  예측값: {y_pred}")
    print(f"  손실(Loss): {loss:.4f}")
    print(f"  평균 기울기 dL/dw: {dL_dw:.4f}, dL/db: {dL_db:.4f}")
    print(f"  업데이트 → w: {w_old:.4f} - {lr}*{dL_dw:.4f} = {w:.4f}")
    print(f"               b: {b_old:.4f} - {lr}*{dL_db:.4f} = {b:.4f}\n")

    # 이력 저장
    history.append({
        'epoch': epoch + 1,
        'w': w,
        'b': b,
        'loss': loss,
        'y_pred_mean': np.mean(y_pred),
        'grad_w': dL_dw,
        'grad_b': dL_db
    })

# 이력 시각화
df = pd.DataFrame(history)
print("[학습 이력]")
print(df)
plt.figure(figsize=(10, 5))
plt.plot(df['epoch'], df['loss'], marker='o', label='Loss')
plt.plot(df['epoch'], df['y_pred_mean'], marker='x', label='Mean Prediction')
plt.axhline(y=np.mean(y_true), color='gray', linestyle='--', label='True Mean y')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Batch Gradient Descent: Loss & Prediction')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()