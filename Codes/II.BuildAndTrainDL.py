# II. Build and train a DL model for price forecasting
from keras.models import Sequential
from keras.layers import Dense

import os
import pandas as pd
import numpy as np
import pickle as pk
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# (a) Load previously built datasets : IL FAUT AJOUTER LE PCA ET LE SCALER
trainset_final = pd.read_csv('./Data/TrainSet_final.csv')
trainset = pd.read_csv('./Data/TrainSet.csv')
testset_final = pd.read_csv('./Data/TestSet_final.csv')
testset = pd.read_csv('./Data/TestSet.csv')

# (b) Build and train several models differentamounts of PCs
for nPCs in [10, 20, 30, 40]:
    print(nPCs)
    X = trainset_final.iloc[:, :nPCs]
    y = trainset["result"]

    # Build model and train it
    classifier = Sequential()
    #First Hidden Layer
    classifier.add(Dense(32, activation='relu', kernel_initializer='random_normal', input_dim=nPCs))
    #Second, third and fourth  hidden Layers
    classifier.add(Dense(32, activation='relu', kernel_initializer='random_normal'))
    classifier.add(Dense(16, activation='relu', kernel_initializer='random_normal'))
    classifier.add(Dense(16, activation='relu', kernel_initializer='random_normal'))

    #Output Layer
    classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))
    #Compiling the neural network
    classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])
    #Fitting the data to the training dataset
    classifier.fit(X,y, batch_size=500, epochs=75, verbose =1)

    pk.dump(classifier, open("./Models/DL_model_{}PC.pkl".format(nPCs),"wb"))


# (c) Test onto the testset : we compare all models and store results in a csv file
accuracies, nPCs_list = [], []
for nPCs in [10, 20, 30, 40]:
    print(nPCs)
    with open("./Models/DL_model_{}PC.pkl".format(nPCs), 'rb') as f:
        clf = pk.load(f)
    # Compute predictions on testset
    preds = (clf.predict(testset_final.iloc[:, :nPCs]) > 0.5)*1

    accuracies.append(np.mean(preds == list(testset['result'])))
    nPCs_list.append(nPCs)

recap = pd.DataFrame({'nPCs' : list(nPCs_list), 'Accuracy' : (list(accuracies))})
recap.to_csv('Comparative_All_models.csv', index = False)

