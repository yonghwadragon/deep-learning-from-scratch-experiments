# 8_activation_effect_in_model.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rc

# 한글 폰트 설정 (NanumGothic 없으면 기본 폰트 사용)
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 깨짐 방지

# 데이터 정의: 비선형 목표 (예: [0, 1, 0])
x = np.array([1.0, 2.0, 3.0])
y_true = np.array([0.0, 1.0, 0.0])

# 활성화 함수 정의
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def tanh(z):
    return np.tanh(z)

def relu(z):
    return np.maximum(0, z)

# 활성화 함수 목록 (선형도 포함)
activations = {
    'None (Linear)': lambda z: z,
    'Sigmoid': sigmoid,
    'Tanh': tanh,
    'ReLU': relu
}

lr = 0.1
epochs = 100

# 결과 저장용 딕셔너리: 각 활성화 함수별로 학습 결과 기록
results = {}

# 각 활성화 함수에 대해 학습 진행
for name, act_fn in activations.items():
    print(f"\n===== 활성화 함수: {name} =====")
    w = 1.0
    b = -2.0
    epoch_list, w_list, b_list, loss_list, y_pred_list = [], [], [], [], []
    
    for epoch in range(epochs):
        # 순전파: z 계산 후 활성화 함수 적용
        z = w * x + b
        y_pred = act_fn(z)
        loss = np.mean((y_true - y_pred) ** 2)
        
        # 역전파: 기울기 계산
        grad_z = -2 * (y_true - y_pred)
        if name == 'Sigmoid':
            grad_act = y_pred * (1 - y_pred)
        elif name == 'Tanh':
            grad_act = 1 - y_pred ** 2
        elif name == 'ReLU':
            grad_act = (z > 0).astype(float)
        else:  # Linear
            grad_act = 1.0
        
        dL_dz = grad_z * grad_act
        dL_dw = np.mean(dL_dz * x)
        dL_db = np.mean(dL_dz)
        
        # 파라미터 업데이트
        w_old, b_old = w, b
        w -= lr * dL_dw
        b -= lr * dL_db
        
        # 각 epoch 결과 저장
        epoch_list.append(epoch + 1)
        w_list.append(w)
        b_list.append(b)
        loss_list.append(loss)
        y_pred_list.append(np.mean(y_pred))
        
        # 터미널에 에포크별 출력 (수식 포함)
        print(f"[Epoch {epoch + 1}]")
        print(f"  y_pred 수식: y_pred = {w_old:.4f} * x + {b_old:.4f} → 활성화 적용 → 평균 예측 = {np.mean(y_pred):.4f}")
        print(f"  손실: loss = ( {y_true} - {np.round(y_pred,4)} )^2 평균 = {loss:.4f}")
        print(f"  기울기: dL/dw = {dL_dw:.4f}, dL/db = {dL_db:.4f}")
        print(f"  업데이트: w: {w_old:.4f} → {w:.4f},  b: {b_old:.4f} → {b:.4f}\n")
    
    # 저장: 활성화 함수별 결과 기록
    results[name] = {
        'epoch': np.array(epoch_list),
        'w': np.array(w_list),
        'b': np.array(b_list),
        'loss': np.array(loss_list),
        'y_pred_mean': np.array(y_pred_list)
    }

# 최종 터미널 출력: 각 활성화 함수별 최종 에포크 결과 요약
print("\n===== 최종 학습 이력 (활성화 함수별) =====")
for name in activations.keys():
    res = results[name]
    print(f"{name}: Epoch {res['epoch'][-1]}, w = {res['w'][-1]:.4f}, b = {res['b'][-1]:.4f}, loss = {res['loss'][-1]:.6f}, 평균 y_pred = {res['y_pred_mean'][-1]:.4f}")

# 시각화: 각 활성화 함수별로 3개의 서브플롯 (행: 활성화 함수, 열: [Loss vs Epoch, Mean Prediction vs Epoch, Final Fit])
num_funcs = len(activations)
fig, axes = plt.subplots(num_funcs, 3, figsize=(15, 4 * num_funcs))

fig.suptitle("활성화 함수별 학습 결과", fontsize=18, y=0.98)

for i, (name, res) in enumerate(results.items()):
    # 서브플롯 1: Loss vs Epoch
    axes[i, 0].plot(res['epoch'], res['loss'], marker='o', color='blue')
    axes[i, 0].set_title(f"{name}: Loss vs Epoch", fontsize=12)
    axes[i, 0].set_xlabel("Epoch", fontsize=10)
    axes[i, 0].set_ylabel("Loss", fontsize=10)
    axes[i, 0].grid(True)
    
    # 서브플롯 2: Mean Prediction vs Epoch
    axes[i, 1].plot(res['epoch'], res['y_pred_mean'], marker='x', color='red')
    axes[i, 1].set_title(f"{name}: Mean Prediction vs Epoch", fontsize=12)
    axes[i, 1].set_xlabel("Epoch", fontsize=10)
    axes[i, 1].set_ylabel("Mean Prediction", fontsize=10)
    axes[i, 1].grid(True)
    
    # 서브플롯 3: Final Fit - 데이터와 최종 예측 직선
    line_x = np.linspace(min(x) - 1, max(x) + 1, 100)
    line_y = res['w'][-1] * line_x + res['b'][-1]
    axes[i, 2].scatter(x, y_true, color='black', label='True Data')
    axes[i, 2].plot(line_x, line_y, color='green', label='Fitted Line')
    axes[i, 2].set_title(f"{name}: Final Fit", fontsize=12)
    axes[i, 2].set_xlabel("x", fontsize=10)
    axes[i, 2].set_ylabel("y", fontsize=10)
    axes[i, 2].legend(fontsize=9, loc='upper left')
    axes[i, 2].grid(True)

# 레이아웃 조정: 타이틀 및 축 레이블 겹침 방지
plt.subplots_adjust(top=0.92, hspace=0.4, wspace=0.3)
plt.show()