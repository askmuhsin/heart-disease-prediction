"""
prediction -->
-- Value 0: < 50% diameter narrowing
-- Value 1: > 50% diameter narrowing
"""
import numpy as np

def getLabel():
    feature_set = {
        1 : "age",
        2 : "sex",
        3 : "chest_pain",
        4 : "resting_bp",
        5 : "chol",
        6 : "fbs",
        7 : "rest_ecg",
        8 : "max_heart_rate",
        9 : "exang",
        10: "oldpeak",
        11: "slope",
        12: "ca",
        13: "thal"
    }
    return feature_set

def getData():
    data = []
    with open('./data/processed.cleveland.data') as f:
        data = f.readlines()
    X = []
    y = []

    for row in data:
        data_1 = row.rstrip()
        data_arr = data_1.split(',')

        feat = data_arr[:len(data_arr)-1]
        X_temp=[]
        for it in feat:
            if it!="?":
                X_temp.append(float(it))
            else:
                X_temp.append(0.0)
        X.append(X_temp)

        y_temp = data_arr[len(data_arr)-1:]
        y_temp = int(y_temp[0])
        y.append(0) if y_temp==0 else y.append(1)

    return X, y

def normalize(feat):
    numerator = feat - min(feat)
    denominator = max(feat)-min(feat)
    z = numerator/denominator
    return z

def getNormData():
    X, y = getData()
    X = np.array(X)
    for i in range(len(X[0])):
        X[:,i] = normalize(X[:, i])
    y = np.array(y)
    y = y.reshape(y.shape[0], 1)
    return X, y

def main():
    X, y = getNormData()
    for x, y_ in zip(X, y):
        print(x, y_, sep='\t')

if __name__ == '__main__':
    main()
