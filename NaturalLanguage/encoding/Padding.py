import torch
from torch.nn.utils.rnn import pad_sequence

# 1. 데이터셋 확장 (연구용 텍스트 예시 포함)
preprocessed_sentences = [
    ["안녕", "하세요"],
    ["반갑습니다"],
    ["안녕", "하십니까"],
    ["지금", "파이토치", "공부", "하세요"],
    ["딥러닝", "자연어", "처리", "입문", "책", "어렵지만", "재밌네요"],
    ["나중에", "비밀", "단어", "찾기", "연구", "할거에요"],
    ["이", "문장은", "엄청나게", "길게", "만들어", "보겠습니다", "패딩이", "어떻게", "들어가는지", "궁금하네요"]
]

# 2. 단어 집합(Vocabulary) 만들기
vocab = {}
all_tokens = [token for sentence in preprocessed_sentences for token in sentence]

for token in all_tokens:
    if token not in vocab:
        # 0은 패딩(PAD)용으로 남겨두고 1부터 인덱스 부여
        vocab[token] = len(vocab) + 1

print("단어 집합 크기:", len(vocab))

# 3. 정수 인코딩 수행
encoded = []
for sentence in preprocessed_sentences:
    current_sentence = [vocab[token] for token in sentence]
    # pad_sequence를 쓰려면 각 문장을 텐서(Tensor) 형태로 바꿔야 합니다.
    encoded.append(torch.tensor(current_sentence))

# 4. 패딩 추가 (pad_sequence)
# batch_first=True: [문장 개수, 최대 길이] 형태로 만듭니다.
# padding_value=0: 빈 공간을 0으로 채웁니다.
padded_encoded = pad_sequence(encoded, batch_first=True, padding_value=0)

print("\n--- 패딩 결과 (행렬) ---")
print(padded_encoded)
print("\n최종 데이터 모양(Shape):", padded_encoded.shape)