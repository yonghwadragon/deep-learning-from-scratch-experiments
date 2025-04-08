# 9_hidden_layer_effect.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rc

plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 깨짐 방지

# -------------------------------------------------
# (1) 환경 설정 및 데이터 정의
# -------------------------------------------------
np.random.seed(42)  # 난수 시드 설정 (재현성)

# 입력과 정답: (3,) 형태
x = np.array([1.0, 2.0, 3.0])
y_true = np.array([0.0, 1.0, 0.0])

# 벡터 연산을 위해 (샘플 수, 특성 수) 형태로 변환
# 여기서는 입력이 스칼라이므로 shape = (3,1)
X = x.reshape(-1, 1)      # (3,1)
Y = y_true.reshape(-1, 1) # (3,1)

# -------------------------------------------------
# (2) 활성화 함수 및 파라미터 초기화
# -------------------------------------------------
def tanh(x):
    """하이퍼볼릭 탄젠트 활성화 함수"""
    return np.tanh(x)

def tanh_derivative(x):
    """tanh의 도함수 = 1 - tanh^2(x)"""
    return 1 - np.tanh(x) ** 2

# 은닉층 2개 노드, 출력층 1개 노드 가정
# w1: (입력1→은닉2), b1: (1,2)
# w2: (은닉2→출력1), b2: (1,1)
w1 = np.random.randn(1, 2) * 0.5
b1 = np.zeros((1, 2))
w2 = np.random.randn(2, 1) * 0.5
b2 = np.zeros((1, 1))

# -------------------------------------------------
# (3) 하이퍼파라미터 설정
# -------------------------------------------------
lr = 0.1      # 학습률
epochs = 1000 # 총 학습 반복 횟수

# -------------------------------------------------
# (4) 학습 이력 기록용 자료구조
# -------------------------------------------------
history = {
    "epoch": [],
    "loss": [],
    "y_preds": []  # 각 에포크 예측값 목록(샘플별)
}

# -------------------------------------------------
# (5) 학습 루프 (벡터 연산)
# -------------------------------------------------
n = len(X)  # 샘플 수 (여기서는 3)

for epoch in range(1, epochs+1):
    # 순전파
    z1 = X @ w1 + b1          # (3,2)
    a1 = tanh(z1)             # (3,2)
    z2 = a1 @ w2 + b2         # (3,1)
    y_pred = z2               # 출력층은 선형 (활성화 X)

    # MSE 손실 계산
    loss = np.mean((Y - y_pred)**2)

    # 역전파
    # 출력층에서의 미분
    # (평균 MSE를 기준으로 하면 1/n을 곱해서 gradient를 구하는 경우도 있으나,
    # 여기서는 간단히 2*(pred - true)의 형태를 사용 후, 학습률에서 보정.)
    dL_dz2 = 2 * (y_pred - Y)        # (3,1)

    # 가중치/편향 w2, b2에 대한 그래디언트
    # a1.T: (2,3), dL_dz2: (3,1) -> 결과 (2,1)
    dL_dw2 = a1.T @ dL_dz2
    dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)  # (1,1)

    # 은닉층 a1에 대한 미분
    dL_da1 = dL_dz2 @ w2.T           # (3,2)
    dL_dz1 = dL_da1 * tanh_derivative(z1) # (3,2)

    # w1, b1에 대한 그래디언트
    # X.T: (1,3), dL_dz1: (3,2) -> 결과 (1,2)
    dL_dw1 = X.T @ dL_dz1
    dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)  # (1,2)

    # 파라미터 업데이트
    # (배치 크기(n=3)에 따라 gradient가 스케일링될 수 있어 필요에 따라 조정 가능)
    w2 -= lr * dL_dw2
    b2 -= lr * dL_db2
    w1 -= lr * dL_dw1
    b1 -= lr * dL_db1

    # 에포크 기록
    history["epoch"].append(epoch)
    history["loss"].append(loss)
    # 현재 에포크의 각 샘플별 예측값(1차원 리스트)
    history["y_preds"].append(list(y_pred.flatten()))

# -------------------------------------------------
# (6) 최종 예측값 및 학습 이력 결과 확인
# -------------------------------------------------
df = pd.DataFrame({"epoch": history["epoch"], "loss": history["loss"]})
print("[학습 이력]")
print(df)

# 최종 예측 확인
final_preds = []
for xi in X:
    z1 = xi @ w1 + b1
    a1 = tanh(z1)
    z2 = a1 @ w2 + b2
    final_preds.append(z2[0,0])

print("\n[최종 예측 결과]")
for i in range(n):
    print(f"x = {X[i,0]:.1f}, 예측값 = {final_preds[i]:.4f}, 정답 = {Y[i,0]:.1f}")

# -------------------------------------------------
# (7) 시각화
# -------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1) 학습 손실 변화
axes[0].plot(history["epoch"], history["loss"], marker='o', linestyle='-')
axes[0].set_title("학습 손실 변화")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("MSE Loss")
axes[0].grid(True)

# 2) 각 샘플의 예측값 변화
for i, xi in enumerate(x):
    preds_per_epoch = [epoch_preds[i] for epoch_preds in history["y_preds"]]
    axes[1].plot(history["epoch"], preds_per_epoch, marker='x', linestyle='-', label=f"x={xi:.1f}")

# 실제 정답 수평선 표시
for i, true_val in enumerate(y_true):
    axes[1].hlines(true_val, xmin=1, xmax=epochs, linestyles='--')

axes[1].set_title("각 샘플 예측값 변화")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("예측값")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()