# 12_1_xor_activation_boundary_numpy.py
import numpy as np
import matplotlib.pyplot as plt

# 한글 폰트 설정 (NanumGothic 없으면 기본 폰트 사용)
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 깨짐 방지

# 1. XOR 입력과 정답 정의
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
Y = np.array([[0], [1], [1], [0]])  # XOR 출력

# 2. 활성화 함수 정의
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1 - np.tanh(x) ** 2

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

# 결정 경계 시각화 함수
def plot_decision_boundary(model_fn, title, X, Y):
    h = 0.01
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model_fn(grid)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.6)
    plt.scatter(X[:, 0], X[:, 1], c=Y.flatten(), cmap=plt.cm.RdBu, edgecolors='k')
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True)
    plt.show()

# 3. 학습 설정
epochs = 10000
lr = 0.1
activation_functions = {
    "Sigmoid": (sigmoid, sigmoid_deriv),
    "Tanh": (tanh, tanh_deriv),
    "ReLU": (relu, relu_deriv)
}

results = {}

# 4. 각 활성화 함수로 학습
for name, (act_fn, act_deriv) in activation_functions.items():
    np.random.seed(0)
    input_size, hidden_size, output_size = 2, 2, 1

    # 가중치 및 편향 초기화
    W1 = np.random.randn(input_size, hidden_size)
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size)
    b2 = np.zeros((1, output_size))

    losses = []
    preds_per_epoch = []

    for epoch in range(epochs):
        # 순전파
        Z1 = X @ W1 + b1
        A1 = act_fn(Z1)
        Z2 = A1 @ W2 + b2
        A2 = sigmoid(Z2)  # 출력층은 고정

        # 손실 계산 (MSE)
        loss = np.mean((Y - A2) ** 2)
        losses.append(loss)
        preds_per_epoch.append(A2.copy())

        # 역전파
        dA2 = (A2 - Y)
        dZ2 = dA2 * sigmoid_deriv(Z2)
        dW2 = A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = dZ2 @ W2.T
        dZ1 = dA1 * act_deriv(Z1)
        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # 가중치 업데이트
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

    # 예측 결과 저장
    final_pred = (A2 > 0.5).astype(int)
    acc = np.mean(final_pred == Y)

    def predict_fn(x):
        Z1 = x @ W1 + b1
        A1 = act_fn(Z1)
        Z2 = A1 @ W2 + b2
        A2 = sigmoid(Z2)
        return (A2 > 0.5).astype(int)

    results[name] = {
        "losses": losses,
        "final_pred": final_pred,
        "acc": acc,
        "probs": A2,
        "predict_fn": predict_fn
    }

# 5. 손실 시각화
plt.figure(figsize=(10, 5))
for name in results:
    plt.plot(results[name]['losses'], label=f"{name}")
plt.title("XOR 문제 - 활성화 함수별 손실 감소")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 6. 결과 출력 및 결정 경계 시각화
for name in results:
    print(f"\n\U0001F50E {name} 결과")
    print("예측 확률:\n", np.round(results[name]['probs'], 3))
    print("예측 클래스:\n", results[name]['final_pred'].flatten())
    print("정확도:", results[name]['acc'])
    plot_decision_boundary(results[name]['predict_fn'], f"{name} 결정 경계", X, Y)