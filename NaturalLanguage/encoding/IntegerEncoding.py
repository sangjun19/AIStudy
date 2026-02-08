from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
# nltk.download('punkt_tab')
# nltk.download('stopwords')

raw_text= "A barber is a person. a barber is good person. a barber is huge person. he Knew A Secret! The Secret He Kept is huge secret. Huge secret. His barber kept his word. a barber kept his word. His barber kept his secret. But keeping and keeping such a huge secret to himself was driving the barber crazy. the barber went up a huge mountain."

sentences= sent_tokenize(raw_text) # Split text into sentences
# print(sentences)

vcoab = {}
preprocessed_senteces = []
stop_words = set(stopwords.words('english'))

for sentence in sentences:
    tokenized_sentence = word_tokenize(sentence.lower()) # Tokenize and convert to lower case
    result = []
    
    for word in tokenized_sentence:
        word = word.lower()
        if word not in stop_words:
            if len(word) > 2:
                result.append(word)
                
                if word not in vcoab:
                    vcoab[word] = 0
                vcoab[word] += 1
    preprocessed_senteces.append(result)
# print(preprocessed_senteces)

# print(vcoab)
vocab_sorted = sorted(vcoab.items(), key = lambda x:x[1], reverse=True)
# print(vocab_sorted)

word_to_index = {}
for i in range(len(vocab_sorted)):
    if vocab_sorted[i][1] < 2:
        continue
    word = vocab_sorted[i][0]
    word_to_index[word] = i+1
# print(word_to_index)

vocab_size = 5
words_frequency = [word for word, index in word_to_index.items() if index >= vocab_size + 1]

for w in words_frequency:
    del word_to_index[w]
# print(word_to_index)

word_to_index['OOV'] = len(word_to_index) + 1
# print(word_to_index)

encoded_sentences = []
for sentece in preprocessed_senteces:
    encoded_sentence = []
    for word in sentece:
        try:
            encoded_sentence.append(word_to_index[word])
        except KeyError:
            encoded_sentence.append(word_to_index['OOV'])
    encoded_sentences.append(encoded_sentence)
print(encoded_sentences)