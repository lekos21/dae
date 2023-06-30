"""
ATTENTION: THIS FUNCTION IS CALIBRATED WITH 200 FEATURES, KNOWING THAT JUST 10 
ARE THE IMPORTANT ONES. THE SHRINKING OF THE NEURONS IS CALIBRATED ON
THESE ASSUMPTION! PLEASE REVIEW THE LAYER SHRINK PERCENTAGE BASED ON YOUR DATA.

THE WEIGHTS ARE ALSO CALIBRATED ON MY SPECIFIC APPLICATION, FEEL FREE TO EDIT 
THEM TO MATCH YOUR PROBLEM.

@author: lekos
"""

import numpy as np
import os
from random import choices


def random(n_features):
    
    #retrieve the number of features
    
    
    
    # for some hyperparameter, define a probability (weight) and choose random 
    batch_size_range = [64, 128, 256, 512]
    batch_size_weight = [0.1, 0.5, 0.5, 0.2]
    batch_size = choices(batch_size_range, batch_size_weight)[0]
    
    layers_number_range = [0, 1, 2, 3, 4, 5]
    layers_number_weight = [0.4, 0.4, 0.2, 0.2, 0.15, 0.1]
    layers_number = choices(layers_number_range, layers_number_weight)
    
    #shrink percentage of neurons between consecutive layers (1 = no shrink)    
    layers_shrink_perc = [.7, .65, .6, .55, .5]
    shrink_weight = [1, 1.2, 1.3, 1.4, 1.3]
    
    
    nodes=[n_features]
    
    for i in range(1, layers_number[0]+1):
        nodes.append(round(nodes[i-1]*choices(layers_shrink_perc, shrink_weight)[0]))
    
    #delete the first layer (n_features)
    nodes.pop(0)
    
    n_epochs = 18000
    
    learning_rate_range=[0.01, 0.005, 0.001]
    learning_rate_weight=[0.2, 0.02, 0.01]
    learning_rate = choices(learning_rate_range, learning_rate_weight)[0]
    #struttura dell'AE, [500, 200] --> input-500-200-central-200-500-input
    #MAX 5 layer
    
    activations = 'sigmoid'        
    patience = 5000
    
    
    
    
    
    hyperparams = [n_epochs, batch_size, activations, learning_rate, nodes, patience]
    
    return hyperparams
