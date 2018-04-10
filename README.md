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
python `data_exp.py`     
This script strips and splits the data into usable vectors.    
Function `getData()` returns the formated data.     

python `svm_heart_disease.py`     
This scripts takes the data and run a Support vector classifier.    
The accuracy is widely varying. At best ~80%+.    

### ToDo
- [ ] construct ANN classifier.
