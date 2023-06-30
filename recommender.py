
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


#CONVERTING DATA INTO ARRAY WITH MOVIES AS COLUMNS AND USERS IN ROWS: USUAL STRUCTURE OF DATA FOR NEURAL NETWORKS & RBMs
def convert(data): #PASS TRAINING AND TEST SET
    #LIST OF LISTS: ONE FOR EACH USER
    new_data = []
    
    for id_users in range(1, nb_users+1):
        
         #ALL IDS OF MOVIES RATED BY THIS USER
        id_movies = data[:, 1][data[:, 0] == id_users]
             
             #ALL RATINGS GIVEN BY THIS USER
        id_ratings = data[:, 2][data[:, 0] == id_users]
             
        ratings = np.zeros(nb_movies)
             #SET THE RATED MOVIES TO THE CORRESPONDING RATING
        ratings[id_movies-1] = id_ratings
             
        new_data.append(list(ratings))
         
    return new_data

training_set = convert(training_set)
test_set = convert(test_set)


