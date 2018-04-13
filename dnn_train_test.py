from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import data_exp
import random

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def feedForward(X_train, w_0, w_1, w_2):
    l1 = sigmoid(np.dot(X_train, w_0))
    l2 = sigmoid(np.dot(l1, w_1))
    l3 = sigmoid(np.dot(l2, w_2))
    return l1, l2, l3

epochs = 200000
mu, sigma = 0, 0.5
gamma = 0.001     #learning rate

random.seed(42)
w_0 = np.random.normal(mu, sigma, (13, 15))
w_1 = np.random.normal(mu, sigma, (15, 7))
w_2 = np.random.normal(mu, sigma, (7, 1))

input, out = data_exp.getNormData()
X_train, X_test, y_train, y_test = train_test_split(input, out,
                                                    test_size=0.33,
                                                    random_state=42)

l1, l2, l3 = feedForward(X_test, w_0, w_1, w_2)
error = y_test-l3
print("Testing set error PRIOR : ", np.mean(np.abs(error)))

error_track = []
for epoch in range(epochs):
    l1, l2, l3 = feedForward(X_train, w_0, w_1, w_2)

    error = y_train - l3
    l3_delta = error*sigmoid_derivative(l3)

    l2_error = np.dot(l3_delta, w_2.T)
    l2_delta = l2_error*sigmoid_derivative(l2)

    l1_error = np.dot(l2_delta, w_1.T)
    l1_delta = l1_error*sigmoid_derivative(l1)

    w_0 += np.dot(X_train.T, l1_delta)*gamma
    w_1 += np.dot(l1.T, l2_delta)*gamma
    w_2 += np.dot(l2.T, l3_delta)*gamma

    if (epoch%1000)==0:
        print("Epoch -> {}, Error -> {:.5}".format(epoch, np.mean(np.abs(error))))
        error_track.append(np.mean(np.abs(error)))

plt.plot(error_track)
plt.title("Training Error")
plt.grid()
plt.show()

l1, l2, l3 = feedForward(X_test, w_0, w_1, w_2)
error = y_test-l3
print("Testing set error : ", np.mean(np.abs(error)))

# for y_, y_hat in zip(y_test, l3):
#     print(*y_, *np.around(y_hat))
