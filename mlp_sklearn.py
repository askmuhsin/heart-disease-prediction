from sklearn.neural_network import MLPClassifier
import numpy as np
import data_exp
import random

X, y = data_exp.getNormData()
y = np.ravel(y)

X_nn, _ = data_exp.getData()

clf = MLPClassifier(solver='lbfgs',
                    alpha=1e-5,
                    hidden_layer_sizes=(15, 2),
                    random_state=1)

clf.fit(X, y)

print("Model Accuracy : ", clf.score(X, y))

##------------------------------------------------------------------------------
label = data_exp.getLabel()
samp_num = random.randint(0, 300)
input_sample = X[samp_num]
input_sample = input_sample.reshape(1, input_sample.shape[0])
print("***************************************************\n")
print("Random sample:")
it = 0;
for key, val in label.items():
    print(val, X_nn[samp_num][it], sep="\t")
    it+=1
print()
print("Model Predicts class : ", *clf.predict(input_sample))
print("Actual output  class : ", y[samp_num])
print("***************************************************")
