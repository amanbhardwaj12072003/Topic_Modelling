from pyexpat import model
import gensim
import os 
from gensim.utils import simple_preprocess
from gensim import corpora
from gensim.test.utils import get_tmpfile
from gensim.corpora import MmCorpus
from gensim import models 
import numpy as np # Importing numpy for making of the matrix 
from gensim.models.phrases import Phrases
from pynvim import encoding 
# import gensim.downloader as api


# Opening a text file 

doc = open('sample3.txt', encoding = 'utf-8')

# Making list of all the words present in the text 

tokenized = []
for sentence in doc.read().split("."):
    tokenized.append(simple_preprocess(sentence , deacc = True))  
print(tokenized)
# deacc = True --> removes accent marks fron the text...

# Now we will be making the dictionary of the tokenized using the corpora.Dictionary() function

my_dictionary = corpora.Dictionary(tokenized)
print(my_dictionary)

# Saving this dictionary as text file 
tmp_fname = get_tmpfile('my_dictionary')
my_dictionary.save_as_text(tmp_fname)
load_dictionary = corpora.Dictionary.load_from_text(tmp_fname)

# Creating bag of words 

BOW_corpus = [my_dictionary.doc2bow(doc , allow_update = True) for doc in tokenized]
print(BOW_corpus)

# Now save and load the corpus 

output_fname = get_tmpfile("BOW_corpus.mm")
# Save corpus to disk
MmCorpus.serialize(output_fname,BOW_corpus)
# Load corpus
load_corpus = MmCorpus(output_fname)

# Creating a TF-IDF matrix in GENSIM

word_weight = []
for doc in BOW_corpus:
    for id,freq in doc:
        word_weight.append([my_dictionary[id] , freq])
print(word_weight)

# Applying TF-IDF model 

tfIdf = models.TfidfModel(BOW_corpus , smartirs = 'ntc')
weight_tfidf = []  # This is the empty list that is going to store the tf-idf score 
for doc in tfIdf[BOW_corpus]:
    for id,  freq in doc:
        weight_tfidf.append([my_dictionary[id] , np.around(freq , decimals=3)])
    
print(weight_tfidf)  # This will print all the word with their TF-IDF scores in a list...

# Creating BIGRAMS
# dataset  = api.load("text8")
dataset = open('sample2.txt',encoding='utf-8')
data = []
for word in dataset:
    data.append(word)

bigram_model = Phrases(data,min_count=3,threshold=10)
print(bigram_model[data[0]])








