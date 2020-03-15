# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import requests
import json# Importing the dataset
dataset = pd.read_csv('./Salary_Data.csv')
print(dataset.head(5))
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values# Splitting the dataset into the Training set and Test set
print("--------------------------")
print(X)
print("--------------------------")
print(y)
print("--------------------------")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)# Fitting Simple Linear Regression to the Training set

print("--------------------------")
print(X_train)
print("--------------------------")
print(X_test)
print("--------------------------")
print(y_train)
print("--------------------------")
print(y_test)
print("--------------------------")

regressor = LinearRegression()
regressor.fit(X_train, y_train)# Predicting the Test set results
y_pred = regressor.predict(X_test)# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[1.8]]))