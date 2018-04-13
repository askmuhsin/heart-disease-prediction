## Using ML to predict heart disease.

### Dataset used:
[Heart disease dataset](http://archive.ics.uci.edu/ml/datasets/heart+disease)     
#### Brief description :     
Feature vector consists of 13 attributes, that shows relevance in predicting the
chance of a person having heart-disease.     
The output is a binary classification as follows :       
prediction -->      
0: < 50%     
1: > 50%   
<br />     
The features are explained in detail here : [Dataset Description](http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/heart-disease.names)        

### Usage
python `dnn_train_test.py`
This script trains a nn with two hidden layers.
Uses `data_exp.py` to get normalized data.

python `data_exp.py`     
This script strips and splits the data into usable vectors.    
Function `getData()` returns the formated data.     

python `svm_heart_disease.py`     
This scripts takes the data and run a Support vector classifier.    
The accuracy is widely varying. At best ~80%+.    

### Result
![training error](https://github.com/askmuhsin/heart-disease-prediction/blob/master/data/error_dnn.png)
