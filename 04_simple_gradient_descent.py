# 04_simple_gradient_descent.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 아주 간단한 모델: y = w * x + b (활성화 함수 없음)
# 목표: 주어진 x, y에 대해 MSE loss 줄이기

# 데이터 정의
x = np.array([1.0])        # 입력값: [1.0]
y_true = np.array([3.0])   # 정답: [3.0]

# 초기 가중치와 편향 (임의 설정)
w = 2.0
b = 0.5

# 학습률
lr = 0.1
print(f"학습률 = {lr}\n")

# 저장할 이력
history = []

# 5번 반복 (epoch)
for epoch in range(5):
    print(f"===== Epoch {epoch + 1} =====")
    print(f"입력값: x = {x}")
    print(f"정답: y_true = {y_true}")
    print(f"초기 가중치: w = {w:.4f}")
    print(f"초기 편향: b = {b:.4f}\n")
    
    # 순전파: 예측값 계산
    y_pred = w * x + b

    # 수식: y_pred = w * x + b
    print(f"예측 수식 계산:")
    print(f"  y_pred = {w:.4f} * {x[0]} + {b:.4f} = {y_pred[0]:.4f}")
    
    # 손실 계산: MSE (평균제곱오차)
    loss = np.mean((y_true - y_pred) ** 2)
    # 수식: loss = (y_true - y_pred)^2
    print(f"\n손실(Loss) 계산:")
    print(f"  loss = ({y_true[0]} - {y_pred[0]:.4f})^2 = {loss:.4f}")
    
    # 미분 (역전파): 가중치, 편향에 대한 미분값
    dL_dw = -2 * x * (y_true - y_pred) # 손실함수를 가중치로 미분
    dL_db = -2 * (y_true - y_pred) # 손실함수를 편향으로 미분
    
    print(f"\n기울기(Gradient) 계산:")
    print(f"  dL/dw = -2 * {x[0]} * ({y_true[0]} - {y_pred[0]:.4f}) = {dL_dw[0]:.4f}")
    print(f"  dL/db = -2 * ({y_true[0]} - {y_pred[0]:.4f}) = {dL_db[0]:.4f}")
    
    # 가중치 업데이트 (경사하강법)
    w_old, b_old = w, b
    w = w - lr * dL_dw[0]
    b = b - lr * dL_db[0]
    
    print(f"\n업데이트 전 가중치와 편향:")
    print(f"  w_old = {w_old:.4f}, b_old = {b_old:.4f}")
    print(f"업데이트 후 (w = w_old - lr * dL/dw, b = b_old - lr * dL/db):")
    print(f"  w = {w_old:.4f} - {lr} * {dL_dw[0]:.4f} = {w:.4f}")
    print(f"  b = {b_old:.4f} - {lr} * {dL_db[0]:.4f} = {b:.4f}\n")
    print("========================================\n")
    
    # 이력 저장
    history.append({
        'epoch': epoch + 1,
        'w': w,
        'b': b,
        'y_pred': y_pred[0],
        'loss': loss,
        'grad_w': dL_dw[0], # 손실함수 L을 가중치 w로 미분한 값 : grad_w
        'grad_b': dL_db[0] # 손실함수 L을 편향 b로 미분한 값 : grad_b
    })

# 표로 출력
df = pd.DataFrame(history)
print("[학습 이력]")
print(df)

# 그래프 출력
plt.figure(figsize=(10, 5))
plt.plot(df['epoch'], df['loss'], marker='o', label='Loss')
plt.plot(df['epoch'], df['y_pred'], marker='x', label='Prediction')
plt.axhline(y=y_true[0], color='gray', linestyle='--', label='True y')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Loss and Prediction over Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
