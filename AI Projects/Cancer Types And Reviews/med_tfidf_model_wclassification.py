# Mehmet VARAN

import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics as ms
from sklearn.model_selection import cross_val_score

#added dat file new columns as disease code and review then read it
data = pd.read_table("train.dat")
data_review = data.iloc[:,1:2]

# convertingt to list
data2 = data_review['Review'].values.tolist()

# removing punctuations
data_cleaned = [headline.lower().translate(str.maketrans('', '', string.punctuation)) for headline in data2]

# applying tfidf
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data_cleaned)

# converting to dataframe
sklearn_tfidf = pd.DataFrame(data = X.toarray(),columns=vectorizer.get_feature_names())

# stemming
stem = PorterStemmer()
data_stemmed = [" ".join([stem.stem(word) for word in sentence.split(" ")]) for sentence in data_cleaned]

# applying tfidf
vectorizer = TfidfVectorizer()
X_stemmed = vectorizer.fit_transform(data_stemmed)

# converting to dataframe
sklearn_tfidf_stemmed = pd.DataFrame(data = X_stemmed.toarray(),columns=vectorizer.get_feature_names())

print(sklearn_tfidf.shape)
print(data.shape)

# splittind data for classification
X_train = sklearn_tfidf.iloc[0:13438,:]
Y_train = data.iloc[0:13438,0:1]
X_test = sklearn_tfidf.iloc[13438:14438,:]
Y_test = data.iloc[13438:14438,0:1]
X_train_stemmed = sklearn_tfidf_stemmed.iloc[0:13438,:]
X_test_stemmed = sklearn_tfidf_stemmed.iloc[13438:14438,:]

# Random forest without stemmed corpus
clf = RandomForestClassifier()
clf.fit(X_train,Y_train.values.ravel())

# Random forest with stemmed corpus
clf2 = RandomForestClassifier()
clf2.fit(X_train_stemmed,Y_train.values.ravel())

# Decision tree without stemmed corpus
dct = DecisionTreeClassifier()
dct.fit(X_train, Y_train.values.ravel())

# Decision tree with stemmed corpus
dct2 = DecisionTreeClassifier()
dct2.fit(X_train_stemmed,Y_train.values.ravel())

# KNN without stemmed corpus
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,Y_train.values.ravel())

# KNN with stemmed corpus
knn2 = KNeighborsClassifier(n_neighbors=3)
knn2.fit(X_train_stemmed,Y_train.values.ravel())

# printing scores
print(clf.score(X_test,Y_test))
print(clf2.score(X_test_stemmed,Y_test))
print(dct.score(X_test,Y_test))
print(dct2.score(X_test_stemmed,Y_test))
print(knn.score(X_test, Y_test))
print(knn2.score(X_test_stemmed, Y_test))

scores_clf = cross_val_score(clf, X_test, Y_test.values.ravel(), cv=10)
scores_clf2 = cross_val_score(clf2, X_test_stemmed, Y_test.values.ravel(), cv=10)
scores_dct = cross_val_score(dct, X_test, Y_test.values.ravel(), cv=10)
scores_dct2 = cross_val_score(dct2, X_test_stemmed, Y_test.values.ravel(), cv=10)
scores_knn = cross_val_score(knn, X_test, Y_test.values.ravel(), cv=10)
scores_knn2 = cross_val_score(knn2, X_test_stemmed, Y_test.values.ravel(), cv=10)

print("Random Forest Without Stemming")
print(scores_clf)
print("Random Forest With Stemming")
print(scores_clf2)

print("Decision Tree Without Stemming")
print(scores_dct)
print("Decision Tree With Stemming")
print(scores_dct2)

print("KNN Without Stemming")
print(scores_knn)
print("KNN With Stemming")
print(scores_knn2)