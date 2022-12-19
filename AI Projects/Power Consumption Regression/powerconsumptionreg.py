# Mehnet VARAN

# -*- coding: utf-8 -*-
"""
@author: mehmet
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('teutancitypowerconsumption.csv')

# veri setinin tanıtılması
print("Veri setinin şekli",data.shape)
print("veri seti değişken tipleri:",data.dtypes)
print("veri setinin ilk 10 satırı")
print(data.head(10))
print("veri setinin istatistiki verileri")
description=data.describe()
print(description)
# veri seti içindeki değişkenlerin dağılımlarının çizdirilmesi
data.hist(bins=10,figsize=(16,9),grid=False);
corr = data.corr(method='pearson')
plt.figure()
sns.heatmap(corr, annot=True)

# sıcaklık değerinin ayrılması
temperature = data.iloc[:,1:2]
# p value kullanarak backward elimination yapılması
import statsmodels.api as sm
X = np.append(arr= np.ones((52416,1)).astype(int), values = data, axis = 1)
X_l = data.iloc[:,[2,3,4,5,6,7,8]].values
X_l = np.array(X_l, dtype=float)
model = sm.OLS(temperature,X_l).fit()
print(model.summary())

# x4 ü denklemden çıkarıp tekrar sorgu yapılması
X = np.append(arr= np.ones((52416,1)).astype(int), values = data, axis = 1)
X_l = data.iloc[:,[2,3,4,6,7,8]].values
X_l = np.array(X_l, dtype=float)
model = sm.OLS(temperature,X_l).fit()
print(model.summary())

# gerekli dataları çekme ve backward elimination sonrası dataları scale etme
x = data.iloc[:,[2,3,4,6,7,8]]
ss = preprocessing.StandardScaler()
rescaledX = ss.fit_transform(x)

# verileri ayırma
x_train, x_test, y_train, y_test = train_test_split(rescaledX,temperature, test_size= 0.20, random_state=37)

# linear regresyon modelini uygulama
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)

# linear regresyon modeli için istatiksel verileri yazdırma
import sklearn.metrics as sm
print("Linear Regression Scores")
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_pred), 2)) 
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_pred), 2)) 
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_pred), 2)) 
print("Explain variance score =", round(sm.explained_variance_score(y_test, y_pred), 2))

# karar ağacı regresyonu uygulama
from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(x_train, y_train)
y_pred = r_dt.predict(x_test)
# karar ağacı için istatiksel verileri yazdırma
print("Decision Tree Regression Scores")
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_pred), 2)) 
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_pred), 2)) 
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_pred), 2)) 
print("Explain variance score =", round(sm.explained_variance_score(y_test, y_pred), 2))









































