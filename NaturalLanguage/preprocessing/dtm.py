from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd

# 문서 내 단어 빈도수

# 1. 예시 데이터 (사용자님의 연구 데이터인 어셈블리 명령어와 유사하게 구성)
corpus = [
    'push eax mov eax 1 pop eax',  # 문서 1
    'mov eax 1 xor eax eax',       # 문서 2
    'push ebx mov ebx 5 pop ebx'   # 문서 3
]

# 2. DTM (CountVectorizer) 생성
cv = CountVectorizer()
dtm_vector = cv.fit_transform(corpus)

# 결과 출력 (데이터프레임 형태)
print("--- DTM 결과 ---")
print(pd.DataFrame(dtm_vector.toarray(), columns=cv.get_feature_names_out()))
print("\n")