import gensim.downloader as api
from gensim.models.phrases import Phrases
from numpy import tri

dataset = api.load("text8")

data = []
for word in dataset:
    data.append(word)

bigram_model = Phrases(data,min_count=3,threshold=10)
# print(bigram_model[data[0]])

# For trigram model we just pass the bigram model as such 

trigram_model = Phrases(bigram_model,threshold=10)
print(trigram_model[bigram_model[data[0]]])

