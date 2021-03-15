from backprop import backprop
from forward import forward
import numpy as np
from numpy import genfromtxt

# fetch data
data = genfromtxt('winequality-red.csv', delimiter=';')

# Standardize data
for i in range(11):
    data[:, i] = (data[:, i] - np.mean(data[1:800, i])) / np.std(data[1:800, i]) + 1

# split test / train
X_train = data[1:801, 0:11]
y_train = data[1:801,[11]]
X_test = data[800:1600, 0:11]
y_test = data[800:1600, [11]]

# train model using backpropogation
w1, w2, error_over_time = backprop(X = X_train, y = y_train, M = 30, iters = 1000, eta = 0.025)

# test model 
Z, y_test_pred = forward(X_test, w1, w2)

# evaluate total accuracy
print(np.sqrt(np.mean(np.square(y_test_pred - y_test))))