'''
Created on Aug 9, 2016
Keras Implementation of Multi-Layer Perceptron (GMF) recommender model in:
He Xiangnan et al. Neural Collaborative Filtering. In WWW 2017.  

@author: Xiangnan He (xiangnanhe@gmail.com)
'''

from time import time
import argparse
import os
import pickle
from tkinter import HIDDEN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import random 

import numpy as np
import visualize
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Input, concatenate, Flatten
from sklearn.metrics import accuracy_score
from evaluate import evaluate_model
from Dataset import Dataset



import torch
import torch.nn as nn
from torch import autograd
from phenotype import FeedForwardNet 




import torch

from neat.population import Population
from neat.visualize import draw_net
from neat.feed_forward import FeedForwardNet
#from tf_neat.neat_reporter import LogReporter



#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run MLP.")
    parser.add_argument('--path', nargs='?', default='Data/Dataset1/', # Path to the datasets we have made
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='new_small_ml-1m', # Change here if you want to use other dataset
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--layers', nargs='?', default='[64,32,16,8]',
                        help="Size of each layer. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument('--reg_layers', nargs='?', default='[0,0,0,0]',
                        help="Regularization for each layer")
    parser.add_argument('--num_neg', type=int, default=4,      # NUMBER OF NEGATIVES INSTANCES 
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    return parser.parse_args()


    #######################################

##Function to define the latent vectors at the beginning of the whole model





class RS:
    # Where to evaluate tensors
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Boolean - print generation stats throughout trial
    VERBOSE = True

    # Number of inputs/outputs each genome should contain
    NUM_INPUTS = 64
    NUM_HIDDENS = 10 
    NUM_OUTPUTS = 1
    # Boolean - use a bias node in each genome
    USE_BIAS = True
    
    # String - which activation function each node will use
    # Note: currently only sigmoid and tanh are available - see v1/activations.py for functions
    ACTIVATION = 'sigmoid'
    # Float - what value to scale the activation function's input by
    # This default value is taken directly from the paper
    SCALE_ACTIVATION = 4.9
    
    # Float - a solution is defined as having a fitness >= this fitness threshold
    FITNESS_THRESHOLD = 0.98

    # Integer - size of population
    POPULATION_SIZE = 10
    # Integer - max number of generations to be run for
    NUMBER_OF_GENERATIONS = 200
    # Float - an organism is said to be in a species if the genome distance to the model genome of a species is <= this speciation threshold
    SPECIATION_THRESHOLD = 3.0

    # Float between 0.0 and 1.0 - rate at which a connection gene will be mutated
    CONNECTION_MUTATION_RATE = 0.80
    # Float between 0.0 and 1.0 - rate at which a connections weight is perturbed (if connection is to be mutated) 
    CONNECTION_PERTURBATION_RATE = 0.90
    # Float between 0.0 and 1.0 - rate at which a node will randomly be added to a genome
    ADD_NODE_MUTATION_RATE = 0.03
    # Float between 0.0 and 1.0 - rate at which a connection will randomly be added to a genome
    ADD_CONNECTION_MUTATION_RATE = 0.5
    
    # Float between 0.0 and 1.0 - rate at which a connection, if disabled, will be re-enabled
    CROSSOVER_REENABLE_CONNECTION_GENE_RATE = 0.25

    # Float between 0.0 and 1.0 - Top percentage of species to be saved before mating
    PERCENTAGE_TO_SAVE = 0.30
    
    # XOR's input and output values
    # Note: it is not always necessary to explicity include these values. Depends on the fitness evaluation.
    # See an OpenAI gym experiment config file for a different fitness evaluation example.
    inputs = []

    targets = []

    def fitness_fn(self, genome):
        fitness = 1.0
        net  = FeedForwardNet(genome, self)
        net.to(self.DEVICE)

        values = []
        gt = []
        i = 0
        for input, target in zip(self.inputs, self.targets):  # 4 training examples
            input, target = input.to(self.DEVICE), target.to(self.DEVICE)
            #print(input)
            #print(target)

            pred = net(input)
            pred = pred.flatten().tolist()
            pred = pred[0]
            if pred >= 0.5:
                pred = 1
            else:
                pred = 0
            gt.append(target)
            values.append(pred)

            i +=1
            
            if i == 100:
                break
    
        fitness = accuracy_score(values, gt)
        if len(set(values)) == 1:
            fitness = 0 
        print(fitness)
        
        return fitness



def get_latent_vectors(num_users, num_items, reg_layers=[0,0,0]):
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    item_input = Input(shape=(1,), dtype='int32', name = 'item_input')

    MLP_Embedding_User = Embedding(input_dim = num_users, output_dim = 32, name = 'user_embedding',
                                   embeddings_initializer='uniform', embeddings_regularizer = l2(reg_layers[0]), input_length=1)
    MLP_Embedding_Item = Embedding(input_dim = num_items, output_dim = 32, name = 'item_embedding',
                                   embeddings_initializer='uniform', embeddings_regularizer = l2(reg_layers[0]), input_length=1)
    
    # Crucial to flatten an embedding vector!
    user_latent = Flatten()(MLP_Embedding_User(user_input))
    item_latent = Flatten()(MLP_Embedding_Item(item_input))
    
    # The 0-th layer is the concatenation of embedding layers
    vector = concatenate([user_latent, item_latent], axis= -1)

    model = Model(inputs=[user_input, item_input],
                  outputs=vector)
    
    return model

##Implemeneting NEAT
def get_model(train, num_users, num_items, num_negatives ,layers = [20,10], reg_layers=[0,0]):

    
    #We need input and output here in order to make_net
    print(train.shape)

    user_train, item_train, labels = get_train_instances(train, num_negatives)
    


    print(len(user_train), len(item_train), len(labels))
    #Define global variables
    get_model.num_user = num_users
    get_model.num_items = num_items
    get_model.user_input = user_train 
    get_model.item_input = item_train
    get_model.labels = labels

    print("a")
    model_emb = get_latent_vectors(get_model.num_user, get_model.num_items)
    #Retrieving the embedding vector for all the inputs
    print("b")
    get_model.embeddings = model_emb.predict([np.array(get_model.user_input),np.array(get_model.item_input)], batch_size=1, verbose=0)

    RS.inputs = list(map(lambda s: autograd.Variable(torch.Tensor([s])), get_model.embeddings))
    RS.targets = list(map(lambda s: autograd.Variable(torch.Tensor([s])), get_model.labels))
    

    neat = Population(RS)
    solution, generation = neat.run()


    print(solution, generation)


    draw_net(solution, view=True, filename='ciao', show_disabled=True)

    exit(0)
    #----- NEAT -------#
    


    return model




def get_train_instances(train, num_negatives):
    user_input, item_input, labels = [],[],[]
    # num_users = train.shape[0]
    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        # negative instances
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in train:
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels

if __name__ == '__main__':
    args = parse_args()
    path = args.path
    dataset = args.dataset
    layers = eval(args.layers)
    reg_layers = eval(args.reg_layers)
    num_negatives = args.num_neg
    learner = args.learner
    learning_rate = args.lr
    batch_size = args.batch_size
    epochs = args.epochs
    verbose = args.verbose
    
    topK = 10
    evaluation_threads = 1 #mp.cpu_count()
    print("MLP arguments: %s " %(args))
    model_out_file = 'Pretrain/%s_MLP_%s_%d.h5' %(args.dataset, args.layers, time())
    
    # Loading data
    t1 = time()
    dataset = Dataset(args.path + args.dataset)
    
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives

    num_users, num_items = train.shape

    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" 
          %(time()-t1, num_users, num_items, train.nnz, len(testRatings)))
    # Build model
    train = train[:50]
    model = get_model(train, num_users, num_items,  num_negatives, layers, reg_layers)
    
    t2 = time()
    print("Execution time :", t2-t1)
    exit(0)

    