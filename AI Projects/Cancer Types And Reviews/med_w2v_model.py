# Mehmet VARAN

# importing libraries
import pandas as pd
import string
from gensim.models import Word2Vec
import nltk
from nltk.stem import PorterStemmer

#added dat file new columns as disease code and review then read it
data = pd.read_table("train.dat")
data_review = data.iloc[:,1:2]

# getting only review raw
data2 = data_review['Review'].values

# removing punctuations
data_cleaned = [headline.lower().translate(str.maketrans('', '', string.punctuation)) for headline in data2]

# tokenizing words 
data_vec = [nltk.word_tokenize(Review) for Review in data_cleaned]

# creating model and saving
model = Word2Vec(data_vec,size=100,window=5,min_count=2,sg=1)

# trying any word
print("Model without stemmed corpus")
print(model.wv['body'])
print(model.wv.most_similar('cancer'))

# stemming
stem = PorterStemmer()
data_stemmed = [" ".join([stem.stem(word) for word in sentence.split(" ")]) for sentence in data_cleaned]

# tokenizing words 
data_vec_stemmed = [nltk.word_tokenize(Review) for Review in data_stemmed]

# creating model and saving
model_stemmed = Word2Vec(data_vec_stemmed,size=100,window=5,min_count=2,sg=1)

# trying any word
print("Model with stemmed corpus")
print(model_stemmed.wv['bodi']) # body becomes bodi after stemmed
print(model_stemmed.wv.most_similar('cancer'))