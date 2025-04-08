import numpy as np 
x = np.linspace(1, 10 , 10)                                
print(x)
x = x.reshape(-1, 1)                       # x를 (100, 1) 형태로 변환
print(x)