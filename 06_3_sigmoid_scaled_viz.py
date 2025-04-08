# 06_3_sigmoid_scaled_viz.py
# 출력층에서 스케일링 적용
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 한글 폰트 설정
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 정의
x = np.array([1.0, 2.0, 3.0])
y_true = np.array([2.0, 4.0, 6.0])

# 시그모이드 함수
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 스케일링 설정 시그모이드의 출력 범위가 [0,1]인데 [2,6]범위를 예측하기 위해 사용.
# y_pred = 4 ⋅ σ(w ⋅ x + b) + 2
scale = 4.0
shift = 2.0

# 초기 파라미터
w = 0.0
b = 0.0

# 하이퍼파라미터
lr = 0.1
epochs = 100

# 기록용
line_history = []
history = []

# 학습
for epoch in range(epochs):
    z = w * x + b
    y_pred = scale * sigmoid(z) + shift
    loss = np.mean((y_true - y_pred) ** 2)

    s = sigmoid(z)
    d_activation = scale * s * (1 - s)
    dL_dw = np.mean(-2 * x * (y_true - y_pred) * d_activation)
    dL_db = np.mean(-2 * (y_true - y_pred) * d_activation)

    w -= lr * dL_dw
    b -= lr * dL_db

    # 예측 곡선 저장 (시각화용)
    line_x = np.linspace(0, 4, 100)
    line_y = scale * sigmoid(w * line_x + b) + shift
    line_history.append((line_x, line_y, f"{epoch + 1}번째 에포크"))

    # 기록
    history.append({
        '에포크': epoch + 1,
        '가중치 w': round(w, 4),
        '편향 b': round(b, 4),
        '손실(MSE)': round(loss, 6),
        '평균 예측값': round(np.mean(y_pred), 4)
    })

# 학습 이력 출력
df = pd.DataFrame(history)
print("[학습 이력]")
print(df)

# 최종 예측 출력
final_preds = scale * sigmoid(w * x + b) + shift
print("\n[최종 예측 결과]")
for i in range(len(x)):
    print(f"x = {x[i]:.1f}, 예측값 = {final_preds[i]:.4f}, 실제값 = {y_true[i]:.1f}")

# -----------------------------
# 예측 곡선 시각화 (10개)
# -----------------------------
plt.figure(figsize=(8, 6))
plt.scatter(x, y_true, color='black', label='실제 데이터 (y = 2x)')

# 총 100 에포크 중 10개 선택하여 시각화
selected_epochs = np.linspace(0, epochs - 1, 10, dtype=int)
for i in selected_epochs:
    lx, ly, label = line_history[i]
    plt.plot(lx, ly, label=label)

plt.title("스케일링된 시그모이드 회귀 예측 곡선")
plt.xlabel("입력 x")
plt.ylabel("예측값 y_pred")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# 손실 및 평균 예측값 변화 시각화
# -----------------------------
plt.figure(figsize=(8, 6))
plt.plot(df['에포크'], df['손실(MSE)'], marker='o', linestyle='-', label='손실(MSE)')
plt.plot(df['에포크'], df['평균 예측값'], marker='x', linestyle='-', label='평균 예측값')
plt.axhline(y=np.mean(y_true), linestyle='--', color='gray', label='정답 평균값')

plt.xlabel("에포크")
plt.ylabel("값")
plt.title("에포크에 따른 손실과 평균 예측값 변화")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()