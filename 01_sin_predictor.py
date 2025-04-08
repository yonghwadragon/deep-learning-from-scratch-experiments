# 01_sin_predictor.py
import numpy as np                # 넘파이 불러오기
import matplotlib.pyplot as plt  # 그래프 출력을 위한 모듈
# from tensorflow.keras.models import Sequential  # 순차적 모델
# from tensorflow.keras.layers import Dense       # 밀집 레이어
from keras.models import Sequential  # 순차적 모델 (권장) # type: ignore
from keras.layers import Dense       # 밀집 레이어 (권장) # type: ignore
from keras import Input

# x = np.linspace(0, 2 * np.pi, 100)         # x 데이터 생성 (0 ~ 2π까지 100개)
# y = np.sin(x)                              # 정답은 sin(x)

x = np.linspace(1, 2, 2)
y = x 

x = x.reshape(-1, 1)                       # x를 (100, 1) 형태로 변환
y = y.reshape(-1, 1)                       # y도 (100, 1) 형태로 변환

model = Sequential()                       # 신경망 모델 생성
model.add(Input(shape=(1,)))               # 명시적 입력층
# model.add(Dense(32, input_dim=1, activation='tanh'))  # 첫 번째 은닉층 ->  위와 아래로 나눔.
model.add(Dense(2, activation='tanh'))    # 첫 번째 은닉층
model.add(Dense(2, activation='tanh'))    # 두 번째 은닉층
model.add(Dense(1))                        # 출력층

model.compile(loss='mse', optimizer='adam')  # 손실함수와 옵티마이저 설정
model.fit(x, y, epochs=2, verbose=0)      # 학습 (1000 -> 2번 반복)

y_pred = model.predict(x)                   # 예측 수행

plt.plot(x, y, label='Real sin(x)')         # 실제 sin(x)
plt.plot(x, y_pred, label='Predicted')      # 예측된 값
plt.legend()                                # 범례 표시
plt.title("sin(x) Prediction with Neural Network")  # 그래프 제목
plt.show()                                  # 그래프 출력