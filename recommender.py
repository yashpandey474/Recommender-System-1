
#IMPORT LIBRARIES
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

#IMPORT THE DATASET
movies = pd.read_csv(
    "ml-1m/movies.dat",
    sep = '::', #SEPARATOR
    header = None,
    engine='python',
    encoding='latin-1'
    )

users = pd.read_csv(
    "ml-1m/users.dat",
    sep = '::', #SEPARATOR
    header = None,
    engine='python',
    encoding='latin-1'
    )

ratings = pd.read_csv(
    "ml-1m/ratings.dat",
    sep = '::', #SEPARATOR
    header = None,
    engine='python',
    encoding = "latin-1"
    )

#PREPARING THE TRAINING AND TESTING SET
training_set = pd.read_csv( 
    'ml-100k/u1.base',
    delimiter = "\t"
    )

#80-20 TRAIN-TEST SPLIT ALREADY EXISTS IN THE DATASET FOLDER

#CONVERT TRAINING SET INTO ARRAY
training_set = np.array(training_set, dtype= 'int')

test_set = pd.read_csv( 
    'ml-100k/u1.test',
    delimiter = "\t"
    )

test_set = np.array(test_set, dtype= 'int')

#GET THE TOTAL NUMBER OF USERS AND MOVIES
nb_users = int(max(max(training_set[:,  0]), max(test_set[:, 0])))
#ERROR FIX
max_training = np.max(training_set[:, 1])
max_test = np.max(test_set[:, 1])
nb_movies = int(max(max_training, max_test))