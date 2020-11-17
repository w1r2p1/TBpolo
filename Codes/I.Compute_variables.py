''' Compute predictive variables from the initial raw dataset, you can then use these variables for price forecasting through ML/DL models (possibly after scaling / PCA / ...)

See https://seb943.github.io/Data/Paper_CreatingATradingBot.pdf for complete report'''


import os
os.chdir("C:\\Users\\Utilisateur\\Desktop\\Crypto\\Codes")


from math import pi
from time import time
from poloniex import Poloniex
import numpy as np
import pandas as pd
from bokeh.plotting import figure, output_file, show
from datetime import date, datetime
import matplotlib.pyplot as plt
import csv
import pickle as pk
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#from sklearn.externals.joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix
import sys
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils.vis_utils import plot_model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense
from keras.utils import np_utils
import sys

#sys.path.extend(['C:\\Users\\Sébastien CARARO\\Desktop\\Crypto\\Codes'])
from functions_crypto import *
from apikeys import *
from trendy import *


pd.options.mode.chained_assignment = None # to avoid chained assignment warning message

pair = 'USDT_BTC'
period = 300

########################### O - DATA AND CONSTANTS ########################
os.chdir("C:\\Users\\Utilisateur\\Desktop\\Crypto\\Data") # set the Data repertory location here

df = pd.read_csv('Dataset_USDT_BTC.csv')
df["date"] = pd.to_datetime(df["date"])
df= df.sort_values(by = 'date')
df.to_csv('RAW_{}_Poloniex_{}.csv'.format(pair, period))
######################## I - COMPUTE VARIABLES ############################
df = df.read_csv('RAW_{}_Poloniex_{}.csv'.format(pair, period))
df = compute_variables1(df) # see functions_crypto.py for details

df.to_csv("Preprocessed_{}_{}.csv".format(pair, period), index = False)
print("Finished computing variables!")