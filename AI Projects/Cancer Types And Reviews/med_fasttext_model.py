# Mehmet VARAN

import pandas as pd
import string
from gensim.models import FastText
from nltk.stem import PorterStemmer


#added dat file new columns as disease code and review then read it
data = pd.read_table("train.dat")
data_review = data.iloc[0:100,1:2]

# convertingt to list
data2 = data_review['Review'].values.tolist()

# removing punctuations
data_cleaned = [headline.lower().translate(str.maketrans('', '', string.punctuation)) for headline in data2]

# creating and saving model
model = FastText(data_cleaned,size=100,window=5,min_count=2,sg=1)

#trying things on model
print("Model without stemmed corpus")
print(model.wv['body'])
print(model.wv.similarity('effect', 'children'))

# stemming
stem = PorterStemmer()
data_stemmed = [" ".join([stem.stem(word) for word in sentence.split(" ")]) for sentence in data_cleaned]

# creating and saving model
model_stemmed = FastText(data_stemmed,size=100,window=5,min_count=2,sg=1)

#trying things on model
print("Model with stemmed corpus")
print(model_stemmed.wv['bodi'])
print(model_stemmed.wv.similarity('effect', 'child'))