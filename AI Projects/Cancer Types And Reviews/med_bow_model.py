# Mehmet VARAN

import pandas as pd
import string
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer


#added dat file new columns as disease code and review then read it
data = pd.read_table("train.dat")
data_review = data.iloc[:,1:2]

# convertingt to list
data2 = data_review['Review'].values.tolist()

# removing punctuations
data_cleaned = [headline.lower().translate(str.maketrans('', '', string.punctuation)) for headline in data2]

# applying count vectorizer function
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data_cleaned)

# converting to dataframe
sklearn_bow = pd.DataFrame(data = X.toarray(),columns=vectorizer.get_feature_names(),index=data_cleaned)

# stemming
stem = PorterStemmer()
data_stemmed = [" ".join([stem.stem(word) for word in sentence.split(" ")]) for sentence in data_cleaned]

# applying count vectorizer function
vectorizer = CountVectorizer()
X_stemmed = vectorizer.fit_transform(data_stemmed)

# converting to dataframe
sklearn_bow_stemmed = pd.DataFrame(data = X_stemmed.toarray(),columns=vectorizer.get_feature_names(),index=data_stemmed)