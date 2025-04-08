# 7_activation_functions.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 깨짐 방지

# 입력값 정의
x = np.linspace(-5, 5, 200)

# 활성화 함수 정의
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

# 출력값 계산
y_sigmoid = sigmoid(x)
y_tanh = tanh(x)
y_relu = relu(x)

# 그래프 출력
plt.figure(figsize=(12, 4))

# sigmoid
plt.subplot(1, 3, 1)
plt.plot(x, y_sigmoid, label='sigmoid', color='blue')
plt.title('Sigmoid Function')
plt.grid(True)
plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)

# tanh
plt.subplot(1, 3, 2)
plt.plot(x, y_tanh, label='tanh', color='green')
plt.title('Tanh Function')
plt.grid(True)
plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)

# relu
plt.subplot(1, 3, 3)
plt.plot(x, y_relu, label='ReLU', color='red')
plt.title('ReLU Function')
plt.grid(True)
plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)

plt.suptitle('활성화 함수 시각화: Sigmoid, Tanh, ReLU')
plt.tight_layout()
plt.show()
