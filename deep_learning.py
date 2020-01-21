import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np

sc = StandardScaler()

data = pd.read_csv('auto-mpg.csv', index_col='car name')
data = data[['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'model year', 'origin']]

data[['cylinders', 'displacement', 'horsepower', 'weight', 'model year', 'origin']] = sc.fit_transform(data[['cylinders', 'displacement', 'horsepower', 'weight', 'model year', 'origin']])

training_set, validation_set = train_test_split(data, test_size = 0.1, random_state = 21)
X_train = training_set.iloc[:,0:-1].values
y_train = training_set.iloc[:,-1].values
X_val = validation_set.iloc[:,0:-1].values
y_val = validation_set.iloc[:,-1].values

model = MLPRegressor(hidden_layer_sizes=(150, 100, 50), max_iter=3000,activation = 'relu',solver='adam',random_state=1)
model.fit(X_train, y_train)


y_pred = model.predict(X_val)

for test, real in zip(y_pred, y_val):
    print(test, real)

def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements

from sklearn.metrics import confusion_matrix
#Comparing the predictions against the actual observations in y_val
cm = confusion_matrix(y_pred, y_val)

#Printing the accuracy
print(accuracy(cm))