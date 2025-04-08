# 12_2_xor_activation_improved_numpy.py
import numpy as np
import matplotlib.pyplot as plt

# 한글 폰트 설정 (NanumGothic 없으면 기본 폰트 사용)
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

# ------------------------------------
# 1. XOR 데이터 정의
# ------------------------------------
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
Y = np.array([[0], [1], [1], [0]])  # XOR 출력

# ------------------------------------
# 2. 활성화 함수 정의
# ------------------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_deriv(z):
    # 시그모이드 미분
    s = sigmoid(z)
    return s * (1 - s)

def tanh(z):
    return np.tanh(z)

def tanh_deriv(z):
    return 1 - np.tanh(z) ** 2

def relu(z):
    return np.maximum(0, z)

def relu_deriv(z):
    return (z > 0).astype(float)

# ------------------------------------
# 2-1. BCE Loss 함수
#    XOR처럼 이진분류 → 시그모이드 출력층 + BCE가 일반적
# ------------------------------------
def bce_loss(A2, Y):
    """
    Binary Cross Entropy
    A2: 예측값(0~1), Y: 정답(0 또는 1)
    """
    eps = 1e-8  # log(0) 방지용
    return -np.mean(Y * np.log(A2 + eps) + (1 - Y) * np.log(1 - A2 + eps))

# ------------------------------------
# 결정 경계 시각화 함수
# ------------------------------------
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

# ------------------------------------
# 3. 설정
# ------------------------------------
epochs = 10000
lr = 0.1
hidden_size = 4  # 12_2에서 2개였는데 4개로 확대
activation_functions = {
    "Sigmoid": (sigmoid, sigmoid_deriv),
    "Tanh": (tanh, tanh_deriv),
    "ReLU": (relu, relu_deriv)
}
results = {}

# ------------------------------------
# 4. 각 활성화 함수로 학습
# ------------------------------------
for name, (act_fn, act_deriv) in activation_functions.items():
    np.random.seed(0)
    input_size, output_size = 2, 1

    # 가중치 및 편향 초기화 (은닉 4개)
    W1 = np.random.randn(input_size, hidden_size) * 0.5
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * 0.5
    b2 = np.zeros((1, output_size))

    losses = []

    for epoch in range(epochs):
        # 순전파
        Z1 = X @ W1 + b1        # (4,4)
        A1 = act_fn(Z1)         # (4,4)
        Z2 = A1 @ W2 + b2       # (4,1)
        A2 = sigmoid(Z2)        # 출력층은 시그모이드

        # 손실 계산 (BCE)
        loss = bce_loss(A2, Y)
        losses.append(loss)

        # 역전파
        # 출력층: Sigmoid + BCE → dZ2 = A2 - Y
        dZ2 = (A2 - Y)
        dW2 = A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        # 은닉층
        dA1 = dZ2 @ W2.T
        dZ1 = dA1 * act_deriv(Z1)
        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # 파라미터 업데이트 (SGD)
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

    # 예측 결과
    final_pred = (A2 > 0.5).astype(int)
    acc = np.mean(final_pred == Y)

    # 결정 경계용 예측 함수
    def predict_fn(x):
        Z1_ = x @ W1 + b1
        A1_ = act_fn(Z1_)
        Z2_ = A1_ @ W2 + b2
        A2_ = sigmoid(Z2_)
        return (A2_ > 0.5).astype(int)

    results[name] = {
        "losses": losses,
        "final_pred": final_pred,
        "acc": acc,
        "probs": A2,
        "predict_fn": predict_fn
    }

# ------------------------------------
# 5. 손실 시각화
# ------------------------------------
plt.figure(figsize=(9, 5))
for name in results:
    plt.plot(results[name]['losses'], label=name)
plt.title("XOR 문제 - 업그레이드 버전 (은닉4 + BCE)")
plt.xlabel("Epoch")
plt.ylabel("BCE Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------------------------
# 6. 결과 출력 + 결정 경계 시각화
# ------------------------------------
for name in results:
    print(f"\n[{name}] 결과")
    print("최종 예측 확률:\n", np.round(results[name]['probs'], 3))
    print("최종 예측 클래스:", results[name]['final_pred'].flatten())
    print("정확도:", results[name]['acc'])
    plot_decision_boundary(results[name]['predict_fn'], f"{name} 결정 경계", X, Y)