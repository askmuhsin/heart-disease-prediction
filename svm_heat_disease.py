import data_exp
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import numpy as np
import time

X, y = data_exp.getData()
X = np.array(X)
y = np.array(y)

rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=rand_state)

t = time.time()
svc = LinearSVC()
svc.fit(X_train, y_train)
print("Time take to train SVC:\t", time.time()-t)

print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

 ## predict performance on random test set
n_predict = 10
t = time.time()
print('My SVC predicts:\t', *svc.predict(X_test[0:n_predict]))
print('For these', n_predict, 'labels: \t', *y_test[0:n_predict])
print(round(time.time()-t, 5), 'Seconds to predict', n_predict,'labels with SVC')
