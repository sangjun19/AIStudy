# 연관이 높으면 1, 낮으면 0으로 표현하는 원-핫 인코딩(One-Hot Encoding)
# 한계: 단어의 유사도를 반영하지 못함, 벡터의 차원이 매우 커짐

from konlpy.tag import Okt

okt = Okt()
tokens = okt.morphs("나는 자연어 처리를 배운다")
print(tokens)

word_to_index = {word: index for index, word in enumerate(tokens)}
print('단어 집합 : ', word_to_index)

def one_hot_encoding(word, word_to_index):
    one_hot_vector = [0]*(len(word_to_index))
    index = word_to_index[word]
    one_hot_vector[index] = 1
    return one_hot_vector

print(one_hot_encoding("자연어", word_to_index))