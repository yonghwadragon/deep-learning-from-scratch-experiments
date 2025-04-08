# 🧐 deep-learning-from-scratch-experiments

개요  
수학적 원리와 수식 중심의 직관적인 신경망 실험 프로젝트입니다.  
NumPy로 원리를 이해하고, PyTorch로 확장하면서 딥러닝 학습 과정을 직관적으로 체험합니다.

---

## 프로젝트 목록

### 1. Sin 함수 예측
- `01_sin_predictor.py`: NumPy 기반 수학 함수 예측
- `01_x_sin_predictor.py`: NumPy 구조 변경 실험
- `02_x_sin_predictor_pytorch.py`: PyTorch MLP로 sin(x) 예측

### 2. 가속 감소(Gradient Descent)
- `03_check.py`: reshape 및 데이터 구조 확인
- `03_x_check.py`: x 구조 확인 전용
- `04_simple_gradient_descent.py`: 단순 GD 수식 실습
- `05_batch_gradient_descent.py`: Batch GD 수식 구현
- `06_1_batch_gradient_descent_visual.py`: GD 시각화 1 (직선 회귀)
- `06_2_batch_gradient_descent_visual.py`: GD 시각화 2 (비교용)
- `06_3_sigmoid_scaled_viz.py`: 시그모이드 + 스케일 조정 시각화

### 3. 활성화 함수(Activation Function) 비교
- `07_activation_functions.py`: sigmoid, tanh, ReLU 시각화 비교
- `08_activation_effect_in_model.py`: 활성화 함수 적용 효과 실험
- `11_activation_advanced_comparison.py`: Swish, GELU 등 고급 함수 비교

### 4. XOR 문제 해결 (MLP 실험)
- `12_1_xor_activation_comparison.py`: NumPy 기반 XOR + MSE + SGD
- `12_2_xor_activation_improved_numpy.py`: NumPy 개선판 (은닉층 증가 + BCE)
- `12_3_xor_activation_comparison.py`: PyTorch 기반 XOR + BCELoss + Adam

### 5. 실제 데이터 예제 & 과적합 방지 기법
- `13_fashion_mnist_multiclass.py`: FashionMNIST 다중 클래스 분류
- `14_california_housing_regression.py`: 캘리포니아 주택 가격 데이터(회귀)
- `15_overfitting_control_methods.py`: Dropout / BatchNorm / L2 정규화로 과적합 억제 (Train vs Test Loss)

---

## 목적
- 수식 가정만으로 MLP의 작동 원리와 학습 과정 이해
- 시각화를 통해 동작 방식 체험
- PyTorch vs NumPy 시간/성능 비교
- 실제 데이터셋(FashionMNIST, CaliforniaHousing) 적용부터 과적합 방지 기법 이해

---

## 사용 기술
- Python 3.x
- NumPy, Matplotlib
- PyTorch
- scikit-learn (회귀 등 실제 데이터셋 전처리)
