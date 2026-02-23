import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# x, y = zip(['a', 1], ['b', 0], ['c', 1], ['d', 0])
# print('X:', x)
# print('y:', y)

# values = [['당신에게 드리는 마지막 혜택!', 1],
# ['내일 뵐 수 있을지 확인 부탁드...', 0],
# ['도연씨. 잘 지내시죠? 오랜만입...', 0],
# ['(광고) AI로 주가를 예측할 수 있다!', 1]]
# columns = ['메일 본문', '스팸 메일 유무']
# df = pd.DataFrame(values, columns=columns)
# print(df)

# np_array = np.array(range(16)).reshape((4, 4))
# print('전체 데이터 : ')
# print(np_array)

# x = np_array[:, :3]
# y = np_array[:, 3]
# print(x)
# print(y)

x, y = np.arange(10).reshape((5, 2)), range(5)

print('X 전체 데이터 : ')
print(x)
print('y 전체 데이터 : ')
print(list(y))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=1234)

print('x 훈련 데이터 : ')
print(x_train)
print('x 테스트 데이터 : ')
print(x_test)

