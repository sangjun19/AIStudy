import numpy as np

# 1. 학습 데이터 (x1, x2)와 정답(target)
# AND 게이트의 데이터셋
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
target = np.array([0, 0, 0, 1])

# 2. 초기값 설정 (가중치와 편향을 0으로 시작)
w = np.array([0.0, 0.0])
b = 0.0
learning_rate = 0.1  # 한 번에 수정할 강도 (보통 작게 설정)

print(f"초기 상태: w={w}, b={b}\n")

# 3. 학습 시작 (Epoch: 전체 데이터를 몇 번 반복해서 볼 것인가)
for epoch in range(10):  # 10번 반복 학습
    print(f"--- Epoch {epoch+1} ---")
    for i in range(len(X)):
        # 현재 w, b로 예측값 계산 (순전파)
        sum_val = np.sum(w * X[i]) + b
        prediction = 1 if sum_val > 0 else 0
        
        # 오차 계산 (정답 - 예측)
        error = target[i] - prediction
        
        # 가중치와 편향 수정 (오차에 비례해서 업데이트)
        # 리버싱으로 치면 "로직 수정 패치"를 자동으로 하는 구간
        w += learning_rate * error * X[i]
        b += learning_rate * error
        
        print(f"입력:{X[i]} 정답:{target[i]} 예측:{prediction} -> 수정된 w:{w}, b:{b:.1f}")
    print()

# 4. 최종 결과 확인
print("학습 완료! 최종 가중치와 편향:")
print(f"w: {w}, b: {b}")