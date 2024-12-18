
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('hiring.csv')

x = dataset.iloc[:, :3]
y = dataset.iloc[:, -1]

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(x, y)

X_train, X_test, Y_train, Y_test = train_test_split(x, y, random_state=0, train_size=0.7)

Y_prediction=regressor.predict(X_test)

r2_score = regressor.score(X_test, Y_test)
print('Accuracy of the model is: ',r2_score)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))