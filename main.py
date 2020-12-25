# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 18:16:33 2020

@author: Alireza

Predicitng wethere an individual makes more than 50K/yr or not.
"""

# Part 1: Importing relevant packages

import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import tensorflow as tf

#------------------------------------------------------------------------------

#Part 2: Function Definition

# Function for converting output to binary class
def output_label(Y):
    
    for i in range(0,len(Y)):
    
      if Y.iloc[i] == ' <=50K' or Y.iloc[i] == ' <=50K.':
         Y.iloc[i] = 0
      if Y.iloc[i] == ' >50K' or Y.iloc[i] == ' >50K.':
         Y.iloc[i] = 1
         
    return(Y)

# Function for categorical variable treatment    
def categorical(X,categorical_index):
 
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()

    for i in categorical_index:
    
        X.iloc[:,i] = le.fit_transform(X.iloc[:,i])    
    
    return(X)

#Function for chekcing balance of output data
    
def balance(Y,string):
    
    count_Y = [(len(Y)-np.sum(Y))/len(Y), np.sum(Y)/len(Y)]
    plt.figure()
    plt.bar(['<=50K','>50K'],height=count_Y,color='g')
    plt.legend()
    plt.xlabel('Class Distribution')
    plt.ylabel('Class')
    plt.title(string)
    
# Function for building and comparing random forest, SVM, and neural network model

def modeling(X_train,Y_train,X_test,Y_test,model_type,svm_type='rbf',input_units=10
             ,optimizer='adam', loss='binary_crossentropy', metrics='accuracy',
             batch_size=100, epochs=32, input_act='relu'
             ,output_act='sigmoid'):

  
  # training and testing a random forset classifier    
  if model_type =='random_forest':

      from sklearn.ensemble import RandomForestClassifier
      clf_rnd = RandomForestClassifier(max_depth=2, random_state=0)
      clf_rnd.fit(X_train, Y_train,sample_weight=None)
      Y_train_pred = clf_rnd.predict(X_train)
      #Y_train_pred = Y_train_pred.tolist()
      Y_test_pred = clf_rnd.predict(X_test)
      #Y_test_pred = Y_test_pred.tolist()
#      f1 = [f1_score(Y_train, Y_train_pred), f1_score(Y_test, Y_test_pred)]
 
    # training and testing an SVM classifier    
  if model_type == 'SVM':
       
      svm_type = 'rbf' # default SVM kernel 
      from sklearn import svm
      clf_svm = svm.SVC(kernel=svm_type) 
      clf_svm.fit(X_train, Y_train) 
      Y_train_pred = clf_svm.predict(X_train)
      Y_train_pred = Y_train_pred.tolist()
      Y_test_pred = clf_svm.predict(X_test)
      Y_test_pred = Y_test_pred.tolist()
#      f1 = [f1_score(Y_train, Y_train_pred), f1_score(Y_test, Y_test_pred)]

  if model_type == 'ANN':
    
     clf_ann = tf.keras.models.Sequential()
     # Adding the input layer and the first hidden layer
     clf_ann.add(tf.keras.layers.Dense(units=input_units, activation=input_act))
     # Adding the second hidden layer
     clf_ann.add(tf.keras.layers.Dense(units=input_units, activation=input_act))
     clf_ann.add(tf.keras.layers.Dense(units=1, activation=output_act))
     clf_ann.compile(optimizer = optimizer, loss = loss, metrics = [metrics])
     clf_ann.fit(X_train, Y_train, batch_size = 32, epochs = 100)
     Y_train_pred = clf_ann.predict(X_train)> 0.5
     Y_train_pred = Y_train_pred.tolist()
     Y_test_pred = clf_ann.predict(X_test) > 0.5
     Y_test_pred = Y_test_pred.tolist()
#     f1 = [f1_score(Y_train, Y_train_pred), f1_score(Y_test, Y_test_pred)]  
 
  return Y_train_pred,Y_test_pred
 
#-----------------------------------------------------------------------------
        
# Part 3: Main script        

features = ['age','workclass','fnlwgt','education','education-num','marital-status',
            'occupation','relationship','race','sex','capital-gain','capital-loss',
            'hours-per-week','native-country','output']

# Reading training data
Data_train = pd.read_csv('adult.DATA',names=features)


# Reading testing data
Data_test = pd.read_csv('testdata.txt',names=features)

# Identofying and removing missing variables in training and testing sets
Data_train  = Data_train.replace(' ?',np.nan)
Data_train = Data_train.dropna()
Data_test = Data_test.replace(' ?',np.nan)
Data_test = Data_test.dropna()

#Splitting data into input output space for both training and testing
X_train = Data_train.iloc[:,:-1]
Y_train = Data_train['output']
Y_train = output_label(Y_train)

X_test = Data_test.iloc[:,:-1]
Y_test = output_label(Data_test.iloc[:,-1])


# Treating caregorical variables
categorical_index = [1,3,4,5,6,7,8,9,13]
X_train = categorical(X_train,categorical_index)
X_test = categorical(X_test,categorical_index)

# Evaluating correlation among numerical features
Numerical_X_train = X_train.drop(X_train.iloc[:,categorical_index],axis=1)
import seaborn as sns
sns.heatmap(Numerical_X_train.corr())

#Create histograms to find how imbalanced the outputs are
balance(Y_train, 'Class Distribution in Training set')
balance(Y_test, 'Class Distribution in test set')

# Preprocessing the data using standard scaler 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler() # Defining the scaler object
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
Y_train = Y_train.to_numpy(dtype='int32')
Y_test = Y_test.to_numpy(dtype='int32')

# Buidling models

# training and testing a random forset classifier
from sklearn.metrics import f1_score
Y_train_rnd_pred, Y_test_rnd_pred = modeling(X_train,Y_train,X_test,Y_test,
                                                      model_type='random_forest')
f1_rnd = [f1_score(Y_train, Y_train_rnd_pred), f1_score(Y_test, Y_test_rnd_pred)]
# training and testing an SVM classifier
Y_train_svm_pred, Y_test_svm_pred = modeling(X_train,Y_train,X_test,Y_test,
                                                      model_type='SVM')
f1_svm = [f1_score(Y_train, Y_train_svm_pred), f1_score(Y_test, Y_test_svm_pred)]  
  
# Training a neural network classifier
[Y_train_ann_pred, Y_test_ann_pred] = modeling(X_train,Y_train,X_test,Y_test,
                                                      model_type='ANN')
f1_ann = [f1_score(Y_train, Y_train_ann_pred), f1_score(Y_test, Y_test_ann_pred)]