from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd

# TF: 문서 내 단어 빈도수
# IDF: 역문서 빈도수, 특정 단어가 여러 문서에 등장할수록 가중치 감소

# 1. 예시 데이터 (사용자님의 연구 데이터인 어셈블리 명령어와 유사하게 구성)
corpus = [
    'push eax mov eax 1 pop eax',  # 문서 1
    'mov eax 1 xor eax eax',       # 문서 2
    'push ebx mov ebx 5 pop ebx'   # 문서 3
]

# 2. TF-IDF 생성
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

# 결과 출력
print("--- TF-IDF 결과 ---")
print(pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out()))