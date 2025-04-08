# 10_training_animation.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.animation as animation
from matplotlib import rc

# =========================
# 1) 데이터 및 초기 설정
# =========================
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(42)

x = np.array([1.0, 2.0, 3.0])
y_true = np.array([0.0, 1.0, 0.0])
X = x.reshape(-1, 1)
Y = y_true.reshape(-1, 1)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

w1 = np.random.randn(1, 2) * 0.5
b1 = np.zeros((1, 2))
w2 = np.random.randn(2, 1) * 0.5
b2 = np.zeros((1, 1))

lr = 0.1
epochs = 1000
history = {"epoch": [], "loss": [], "y_preds": []}

# =========================
# 2) 인터랙티브 모드 설정
# =========================
plt.ion()  # 인터랙티브 모드 ON
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# =========================
# 3) 학습 + 실시간 그래프
# =========================
for epoch in range(1, epochs+1):
    # 순전파
    z1 = X @ w1 + b1
    a1 = tanh(z1)
    z2 = a1 @ w2 + b2
    y_pred = z2

    # 손실 계산
    loss = np.mean((Y - y_pred)**2)

    # 역전파
    dL_dz2 = 2 * (y_pred - Y)
    dL_dw2 = a1.T @ dL_dz2
    dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)
    dL_da1 = dL_dz2 @ w2.T
    dL_dz1 = dL_da1 * tanh_derivative(z1)
    dL_dw1 = X.T @ dL_dz1
    dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)

    w2 -= lr * dL_dw2
    b2 -= lr * dL_db2
    w1 -= lr * dL_dw1
    b1 -= lr * dL_db1

    # 기록
    history["epoch"].append(epoch)
    history["loss"].append(loss)
    history["y_preds"].append(list(y_pred.flatten()))

    # ---- 실시간 그래프 갱신 ----
    axes[0].cla()
    axes[0].plot(history["epoch"], history["loss"], marker='o', linestyle='-')
    axes[0].set_title("학습 손실 변화 (실시간)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].grid(True)

    axes[1].cla()
    for i, xi in enumerate(x):
        preds_per_epoch = [ep_preds[i] for ep_preds in history["y_preds"]]
        axes[1].plot(history["epoch"], preds_per_epoch, marker='x', linestyle='-', label=f"x={xi:.1f}")
    for val in y_true:
        axes[1].hlines(val, xmin=1, xmax=epochs, linestyles='--', colors='gray')
    axes[1].set_title("각 샘플 예측값 변화 (실시간)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("예측값")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.pause(0.05)  # 잠시 대기 (창이 닫히지 않고 업데이트하도록)
    
plt.ioff()  # 실시간 그래프 모드 OFF
plt.show()  # 학습 끝난 뒤 최종 결과 확인

print("=== 학습 종료 ===")
final_preds = history["y_preds"][-1]
for i, xi in enumerate(x):
    print(f"x={xi}, 예측={final_preds[i]:.4f}, 실제={y_true[i]}")

# # =========================
# # 4) FuncAnimation으로 재생
# # =========================

# # (선택) 기존 figure를 닫고 새로 생성
# plt.close(fig)  
# fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

# def update(frame):
#     """
#     frame: 0 ~ epochs-1
#     history에서 frame번째까지의 데이터를 사용해 그래프 갱신
#     """
#     axes2[0].cla()
#     axes2[1].cla()

#     # 앞쪽 frame+1개까지의 데이터
#     ep_range = history["epoch"][:frame+1]
#     loss_vals = history["loss"][:frame+1]
    
#     # 1) 손실 그래프
#     axes2[0].plot(ep_range, loss_vals, marker='o', linestyle='-')
#     axes2[0].set_title("학습 손실 변화 (애니메이션)")
#     axes2[0].set_xlabel("Epoch")
#     axes2[0].set_ylabel("MSE Loss")
#     axes2[0].grid(True)

#     # 2) 각 샘플 예측값
#     for i, xi in enumerate(x):
#         preds_until_now = [history["y_preds"][ep][i] for ep in range(frame+1)]
#         axes2[1].plot(ep_range, preds_until_now, marker='x', linestyle='-', label=f"x={xi:.1f}")
#     for val in y_true:
#         axes2[1].hlines(val, xmin=1, xmax=epochs, linestyles='--', colors='gray')
#     axes2[1].set_title("각 샘플 예측값 변화 (애니메이션)")
#     axes2[1].set_xlabel("Epoch")
#     axes2[1].set_ylabel("예측값")
#     axes2[1].legend()
#     axes2[1].grid(True)

#     plt.tight_layout()

# ani = animation.FuncAnimation(fig2, update, frames=epochs, interval=100, repeat=False)
# plt.show()

# # (원한다면 mp4나 gif로 저장 가능)
# # ani.save('training_animation.mp4', fps=15, extra_args=['-vcodec', 'libx264'])