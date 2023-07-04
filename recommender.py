
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

#CONVERT LIST OF LISTS INTO TORCH TENSORS
#->TENSOR IS A PYTORCH ARRAY; MULTIDIMENSIONAL MATRIX
training_set = torch.FloatTensor(
   training_set
    )

test_set = torch.FloatTensor(
    test_set
  )

#-> BUILDING A RESTRICTED BOLTZMANN MACHINE
#CONVERT RATINGS INTO BINARY: 1 [LIKED] OR 0 [NOT LIKED]  or -1 [NOT RATED]
#1. NO RATING -> -1
training_set[training_set == 0] = -1
#2. NOT LIKED -> 0 [CANNOT USE or OPERATOR FOR TENSORS]
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
#3, LIKED -> 1
training_set[training_set >= 3] = 1

#1. NO RATING -> -1
test_set[test_set == 0] = -1
#2. NOT LIKED -> 0 [CANNOT USE or OPERATOR FOR TENSORS]
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
#3, LIKED -> 1
test_set[test_set >= 3] = 1

#ARCHITECTURE OF RESTRICTED BOLTZMANN MACHINE
class RBM():
    #NV: NUMBER OF VISIBLE NODES. NH: NUMBER OF HIDDEN NODES
    def __init__(self, nv, nh,):
        #INITIALISE RANDOM WEIGHTS OF SIZE (NH, NV) ACCORDING TO NORMAL DISTRIBUTION
        self.W = torch.randn(nh,
                             nv)
        #INITIALIZE THE BIAS: 1 FOR EACH HIDDEN NODE
        self.a = torch.randn(1, 
                             nh)
        #INITIALIZE THE BIAS: 1 FOR EACH VISIBLE NODE
        self.b = torch.randn(1,
                             nv)
    
    #SAMPLING THE HIDDEN NODES WITH A PROBABILITY [SIGMOID] [GIBBS SAMPLING]
    def sample_h(self, x): #X = VECTOR OF VISIBLE NEURON VALUES
            #APPLY SIGMOID FUNCTION: f(Wx + a)
            
            #MULTIPLY TENSORS
            wx = torch.mm(x, self.W.t()) #TRANSPOSE OF WEIGHTS USED
            #ADD TENSORS
            activation_input = wx + self.a.expand_as(wx) #EXTRA DIMENSION FOR BIAS APPLIED TO EACH BATCH
            
            #PROBABILITY OF THE HIDDEN NODE GIVEN VISIBLE NODES USING SIGMOID FUNCTION
            probability_h_given_v = torch.sigmoid(activation_input)
            
            #RETURN THE PROBABILITY OF HIDDEN NODES GIVEN THE VISIBLE NODES AND SAMPLE ACCORDING TO THAT PROBABILITY
            return probability_h_given_v, torch.bernoulli(probability_h_given_v)
        
    #SAMPLING THE VISIBLE NODES GIVEN THE HIDDEN NODES
    def sample_v(self, y):
        #APPLY SIGMOID FUNCTION: f(Wx + a)
        
        #MULTIPLY TENSORS
        wy = torch.mm(y, self.W)
        #ADD TENSORS
        activation_input = wy + self.b.expand_as(wy) #EXTRA DIMENSION FOR BIAS APPLIED TO EACH BATCH
        
        #PROBABILITY OF THE HIDDEN NODE GIVEN VISIBLE NODES USING SIGMOID FUNCTION
        probability_v_given_h = torch.sigmoid(activation_input)
        
        #RETURN THE PROBABILITY OF VISIBLE NODES GIVEN THE HIDDEN NODES AND SAMPLE ACCORDING TO THAT PROBABILITY
        return probability_v_given_h, torch.bernoulli(probability_v_given_h)
    
    #CONTRASTIVE DIVERGENCE SOLUTION TO ESTIMATE THE LIKELIHOOD FUNCTION
   #V0 -> RATINGS BY ONE USER
   #VK -> VISIBLE NODES AFTER K ITERATIONS OF CONTRASTIVE DIVERGENCE
   #PH -> PROBABILITIES OF HIDDEN NODES = 1 GIVEN THE VISIBLE NODES AT BEINNING [0] AND AFTER KTH ITERATION [K]
    def train(self, v0, vk, ph0, phk):
        #ADJUST THE WEIGHTS [ACCORDING TO FORMULA]
        self.W += torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)
        #UPDATE BIAS
        self.b += torch.sum((v0-vk), 0) 
        self.a += torch.sum((ph0-phk), 0) 
        
#CREATE RBM OBJECT [INITIALISATION]
nv = len(training_set[0]) #VISIBLE NODES: NO OF FEATURES
nh = 100 #HIDDEN NODES: TUNABLE
batch_size = 100 #ALSO TUNABLE
rbm = RBM(nv, nh)
        
#TRAINING THE RESTRICTED BOLTZMANN MACHINE
nb_epochs = 10 #TUNABLE
#FOR EACH EPOCH
for epoch in range(1, nb_epochs+1):
    #LOSS FUNCTION [BETWEEN PREDICTIONS AND RATINGS]
    train_loss = 0
    #NORMALISE THE TRAIN LOSS BY DIVIDING
    counter = 0
    #GET BATCHES OF USERS
    for id_user in range(0, nb_users - batch_size, batch_size):
        #INPUTS AND TARGETS
        vk = training_set[id_user: id_user+batch_size] #VK
        #TO BE COMPARED
        v0 = training_set[id_user: id_user+batch_size] #SAME AS INPUT AT START
        #PROBABILITY OF HIDDEN NODES GIVEN THE RATINGS AT BEGINNING
        ph0,_ = rbm.sample_h(v0) #_ TO RETURN FIRST RETURNED ELEMENT
        #PROBABILITY OF HIDDEN NODES GIVEN RATINGS AT KTH ITERATION OF CONTRASTIVE DIVERGENCE
        for k in range(10):
            #SAMPLE HIDDEN NODES
            _,hk = rbm.sample_h(vk)
            #UPDATE VISIBLE NODES
            _,vk = rbm.sample_v(hk)
            #DONT LEARN FOR NO RATING [-1]
            vk[v0<0] = v0[v0<0]
        
        #COMPUTE PHK
        phk,_ = rbm.sample_h(vk)
        #UPDATE WEIGHTS & BIAS
        rbm.train(v0, vk, ph0, phk)
        #UPDATE THE TRAINING LOSS BETWEEN PREDICTIONS AND ACTUAL [MEAN ABSOLUTE ERROR]
        train_loss += torch.mean(torch.abs(vk[v0>0]-v0[v0>0]))
        #UPDATE COUNTER
        counter += 1
        
    print(f"EPOCH {epoch} COMPLETE. LOSS: {train_loss/counter}")
     #MOVE TO NEXT BATCH OF USERS
        
            
#MAKE PREDICTIONS ON TEST SET AND COMPUTE TEST LOSS

#LOSS FUNCTION [BETWEEN PREDICTIONS AND RATINGS]
test_loss = 0
#NORMALISE THE TRAIN LOSS BY DIVIDING
counter = 0
#DO NOT NEED BATCH SIZE FOR TESTING [PREDICTION FOR EACH USER]
for id_user in range(nb_users):
    #INPUTS AND TARGETS [USE TRAINING SET TO PREDICT RATINGS IN TEST SET]
    v = training_set[id_user: id_user+1]
    #TO BE COMPARED
    vt = test_set[id_user: id_user+1] 
    
    #CONTRASTIVE DIVERGENCE: JUST 1 STEP [PRINCIPLE OF BLIND WALK]
    if len(vt[vt>=0]) > 0: #VALID RATINGS EXIST
        #SAMPLE HIDDEN NODES
        _,h = rbm.sample_h(v)
        #UPDATE VISIBLE NODES
        _,v = rbm.sample_v(h)
    
        #UPDATE THE TEST LOSS BETWEEN PREDICTIONS AND ACTUAL [MEAN ABSOLUTE ERROR]
        test_loss += torch.mean(torch.abs(vt[vt>0]-v[vt>0]))
        #UPDATE COUNTER
        counter += 1
        
print(f"TEST LOSS: {test_loss/counter}")
            
        
        
    
