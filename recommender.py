
#IMPORT LIBRARIES
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable


#PART1 - DATA PREPROCESSING

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

#CONVERT TRAINING SET INTO ARRAY
training_set = np.array(training_set, dtype= 'int')

test_set = pd.read_csv( 
    'ml-100k/u1.test',
    delimiter = "\t"
    )

test_set = np.array(test_set, dtype= 'int')

#GET THE TOTAL NUMBER OF USERS AND MOVIES
nb_users = int(max(max(training_set[:,  0]), max(test_set[:, 0])))
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

#CONVERT LIST OF LISTS INTO TORCH TENSORS
#->TENSOR IS A PYTORCH ARRAY; MULTIDIMENSIONAL MATRIX
training_set = torch.FloatTensor(
   training_set
    )

test_set = torch.FloatTensor(
    test_set
  )

#PART-2: CREATING THE STACKED AUTOENCODER
class SAE(nn.Module): #INHERITANCE
    #MANDATORY, INITIALISATION FUNCTION
    def __init__(self, ):
        super(SAE, self).__init__() #CLASS AND OBJECT: GET THE METHODS OF PARENT CLASS
        
        #FIRST FULL CONNECTION #ENCODING - 1
        self.fc1 = nn.Linear(
            nb_movies, #NO OF FEATURES
            20 #NO OF NEURONS IN FIRST HIDDEN LAYER: TUNABLE
            )
        #SECCOND FULL CONNECTON #ENCODING - 2
        self.fc2 = nn.Linear(
            20, #NO OF NEURONS IN PREVIOUS LAYER
            10 #NO OF NEURONS IN CURRENT LAYER
            )
        #THIRD FULL CONNECTION #DECODING -1
        self.fc3 = nn.Linear(
            10,
            20
            )
        #FOURTH FULL CONNECTION: #DECODING - 2
        self.fc4 = nn.Linear( #OUTPUT LAYER
            20,
            nb_movies
            )
        #ACTIVATION FUNCTION
        self.activation_function = nn.Sigmoid()
    
    #ENCODING AND DECODING FUNCTION [FORWARD]
    def forward(self, input_features):
        #FIRST ENCODING
        input_features = self.activation_function(
            self.fc1(input_features)
        )
        
        #SECOND ENCODING
        input_features = self.activation_function(
            self.fc2(input_features)
        )
        #FIRST DECODING
        input_features = self.activation_function(
            self.fc3(input_features)
        )
        #SECOND DECODING
        input_features = ( #NO ACTIVATION FUNCTION FOR OUTPUT [SPECIFICITY OF AUTOENCODERS]
            self.fc4(input_features)
        )
        return input_features
    
#INITIALISE THE AUTOENCODER
sae = SAE()
criterion_loss  = nn.MSELoss()
optimizer = optim.RMSprop(
    sae.parameters(),
    lr = 0.01, #LEARNING RATE: TUNABLE
    weight_decay = 0.5 #TUNABLE
    )
#TRAINING THE AUTOENCODER
number_epochs  = 200 #TUNABLE
for epoch in range(1, number_epochs+1):
    train_loss = 0
    user_rated_atleast_1 = 0.
    #LOOP OVER ALL USERS
    for id_user in range(nb_users):
        
        #INPUT VECTOR OF RATINGS GIVEN BY THIS USER: ADD NEW DIMENSION FOR BATCH
        input_features = Variable(training_set[id_user]).unsqueeze(0) #BATCH OF 1 INPUT VECTOR
        #TARGET VECTOR [ORIGINAL INPUT BEFORE ENCODING/DECODING]
        target = input_features.clone()
        
        #ONLY USERS WHO RATED ATLEAST 1 MOVIE
        if torch.sum(target.data>0)>0:
            
            #GET THE PREDICTIONS
            predicted_ratings = sae.forward(input_features)
            #STOCHASTIC GRADIENT DESCEND
            target.require_grad = False #NOT COMPUTE GRADIENT WRT TARGET [OPTIMIZE CODE]
            #NON-LIKED PREDICTIONS
            predicted_ratings[target == 0] = 0 #DONT COUNT IN UPDATING WEIGHTS SO NO AFFECT
            #COMPUTE THE LOSS
            user_loss = criterion_loss(predicted_ratings, target)
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10) #1E-10: TO NEVER MAKE DENOMINATOR ZERO
            #BACKWARD METHOD: DETEREMINE IF TO INCREASE OR DECREASE WEIGHTS [DIRECTION]
            user_loss.backward()
            train_loss += np.sqrt(user_loss.item()*mean_corrector) #MSE & CORRECTOR FOR WHERE NO RATING PREDICTIONS
            user_rated_atleast_1 += 1.
            #OPTIMIZER TO UPDATE THE WEIGHTS [INTENSITY OF UPDATION]
            optimizer.step()
            
    print(f"EPOCH {epoch} COMPLETED. TRAINING LOSS = {train_loss/user_rated_atleast_1}")

#TEST THE AUTOENCODER
test_loss = 0
user_rated_atleast_1 = 0.
for id_user in range(nb_users):
    input_features = Variable(training_set[id_user]).unsqueeze(0) #BATCH OF 1 INPUT VECTOR
    target = Variable(test_set[id_user]).unsqueeze(0)
    
    if torch.sum(target.data>0)>0:
        predicted_ratings = sae.forward(input_features)
        target.require_grad = False
        predicted_ratings[target == 0] = 0
        user_loss = criterion_loss(predicted_ratings, target)
        mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10) #1E-10: TO NEVER MAKE DENOMINATOR ZERO
        test_loss += np.sqrt(user_loss.item()*mean_corrector) #MSE & CORRECTOR FOR WHERE NO RATING PREDICTIONS
        user_rated_atleast_1 += 1.

print(f"TEST LOSS = {test_loss/user_rated_atleast_1}")

        