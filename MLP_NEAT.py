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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import random 

import numpy as np
import neat
import visualize
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Input, concatenate, Flatten
from sklearn.metrics import accuracy_score
from evaluate import evaluate_model
from Dataset import Dataset


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

    n_generations = 150
    config_path = os.path.join(os.path.dirname(__file__), "neat.cfg") # CHANGE YOUR FILENAME WITH YOUR NAME
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )
    
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

    #----- NEAT -------#
    
    
    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    reporter = neat.StdOutReporter(True)
    pop.add_reporter(reporter)

    #Stats
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    #eval_genomes has to be the fitness function
    model = pop.run(eval_genomes, n_generations)

    #Save model as pickle file
    with open('winner.pkl', 'wb') as output:
        pickle.dump(model, output, 1)
    
    print("Best model", model, type(model))


    # Plots the evolution of the best/average fitness
    visualize.plot_stats(stats, ylog=True)
    # Visualizes speciation
    visualize.plot_species(stats)
    # visualize the best topology
    visualize.draw_net(config, model, view=True)

    return model


def eval_genomes(genomes, config):
    #SETTING THE EMBEDDING / LATENT VECOTRS
    # index = [i for i in range(len(embeddings))]
    # keep_index = random.choice(index, k=100)

    print('Output', len(get_model.embeddings), len(get_model.embeddings[0]))
    print('NNS',len(genomes))
    for _, genome in genomes:
            values, gt = [], []
            genome.fitness = len(get_model.embeddings[0])
            for i in range(len(get_model.embeddings)):
            # for i in keep_index: 
                net = neat.nn.FeedForwardNetwork.create(genome, config)
                result = net.activate(get_model.embeddings[i])
                if result[0] >= 0.5:
                    result = 1
                else:
                    result = 0
                values.append(result)
                gt.append(get_model.labels[i])
                '''
                if i == 100:
                    break
                '''
            acc = accuracy_score(values, gt)
            if len(set(values)) == 1:
                genome.fitness = 0 
            else:
                genome.fitness = acc
            print(genome.fitness)

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
    #train = train[:200][:200]
    model = get_model(train, num_users, num_items,  num_negatives, layers, reg_layers)
    
    t2 = time()
    print("Execution time :", t2-t1)
    exit(0)

    