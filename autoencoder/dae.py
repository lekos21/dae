# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 15:15:43 2020

@author: Alex
"""


import time
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
from keras.optimizers import Adam
from autoencoder import dae
import pickle




def dae_build(shapes_path, hyperparams, nodes_central):
    
      

    n_epochs = hyperparams[0]
    batch_size = hyperparams[1]
    activations = hyperparams[2]
    learning_rate = hyperparams[3]
    nodes = hyperparams[4]      
    patience = hyperparams[5]
    
    
    ### OTHER ###
    
    optimizer = Adam(lr=learning_rate)
    
    #define callbacks
    my_callbacks = [ 
        EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience),
        ModelCheckpoint('./DAE_checkpoints/model_best_'+ str(nodes) +'_'+ str(nodes_central) 
                        +'.hdf5', save_best_only=True, mode='min' )]

    #import the shapes
    with open(shapes_path, 'r') as f:
        dataset = np.array([[float(num) for num in line.split(',')] for line in f])
    
    # Subtract the mean shape
    mean_shape=[]
    for k in range(len(dataset[:,0])):
        mean_shape.append(np.mean(dataset[k,:]))            #mean shape
    mean_shape=np.array(mean_shape)
    for k in range(len(dataset[0,:])):
        dataset[:,k] = dataset[:,k]-mean_shape              #subtract it to each centroid
    
    
    #retrieve original variance and mean shape
    with open('./output/general_datas.pkl', 'rb') as f:  
        var_orig, mean_shape = pickle.load(f) 
    
    #transpose
    dataset = np.transpose(dataset)
    
    #Normalization of dataset
    scaler = StandardScaler()
    dataset = scaler.fit_transform(dataset)
    
    
    
    # ------------------------ CROSSVALIDATION SPLIT -----------------------------
    
    
    train_index_h=[]
    # This create X_test and the new dataset: n_splits=10 --> (train+val)/test=90/10
    X_test = dataset[:round(len(dataset)/10),:]
    dataset_no_test = dataset[round(len(dataset)/10):,:]
    
    # This create training and validation set: n_splits=5 --> train/val=80/20
    X_train = dataset_no_test[:round(len(dataset_no_test)*8/10),:]
    X_val = dataset_no_test[round(len(dataset_no_test)*8/10):,:]
    
    
    
    
    # ---------------------- STRUTTURA DELL'AUTOENCODER --------------------------
    
        
        
        #Gli if servono a modificare la struttura dell'AE
    input_img = Input(shape=X_train[0,:].shape)
    if len(nodes)>=0:
        encoded = input_img
    if len(nodes)>=1:
        encoded = layers.Dense(units=nodes[0], activation=activations)(encoded)
    if len(nodes)>=2:
        encoded = layers.Dense(units=nodes[1], activation=activations)(encoded)
    if len(nodes)>=3:
        encoded = layers.Dense(units=nodes[2], activation=activations)(encoded)
    if len(nodes)>=4:
        encoded = layers.Dense(units=nodes[3], activation=activations)(encoded)
    if len(nodes)>=5:
        encoded = layers.Dense(units=nodes[4], activation=activations)(encoded)    
    
    #Central layer
    central = layers.Dense(units=nodes_central, activation=activations)(encoded)
    if len(nodes)==5:
        decoded = layers.Dense(units=nodes[4], activation=activations)(central)
        decoded = layers.Dense(units=nodes[3], activation=activations)(decoded)
        decoded = layers.Dense(units=nodes[2], activation=activations)(decoded)
        decoded = layers.Dense(units=nodes[1], activation=activations)(decoded)
        decoded = layers.Dense(units=nodes[0], activation=activations)(decoded)
    
    if len(nodes)==4:
        decoded = layers.Dense(units=nodes[3], activation=activations)(central)
        decoded = layers.Dense(units=nodes[2], activation=activations)(decoded)
        decoded = layers.Dense(units=nodes[1], activation=activations)(decoded)
        decoded = layers.Dense(units=nodes[0], activation=activations)(decoded)
        
    if len(nodes)==3:
        decoded = layers.Dense(units=nodes[2], activation=activations)(central)
        decoded = layers.Dense(units=nodes[1], activation=activations)(decoded)
        decoded = layers.Dense(units=nodes[0], activation=activations)(decoded)
        
    if len(nodes)==2:
        decoded = layers.Dense(units=nodes[1], activation=activations)(central)
        decoded = layers.Dense(units=nodes[0], activation=activations)(decoded)
        
    if len(nodes)==1:
        decoded = layers.Dense(units=nodes[0], activation=activations)(central)
    
    if len(nodes)==0:
        decoded = layers.Dense(units=len(X_train[0,:]), activation='linear')(central)
    else:
        decoded = layers.Dense(units=len(X_train[0,:]), activation='linear')(decoded)
    
    
    
    #encoded= layers.Dropout(0.2)(encoded)
    
    #autoencoder.summary()
    encoder = Model(input_img, encoded)
    autoencoder=Model(input_img, decoded)
    
    autoencoder.compile(optimizer=optimizer, loss='mse')
    
    #Lancio l'autoencoder per il fit
    track = autoencoder.fit(X_train, X_train,
                    epochs=n_epochs,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(X_val, X_val),
                    callbacks=my_callbacks
                    )
    
    
    #carico il modello che ha ottenuto lo score migliore
    autoencoder.load_weights('./DAE_checkpoints/model_best_'+ str(nodes) +'_'
                             + str(nodes_central) +'.hdf5')
    
    #predizioni del dataset codificato e decodificato
    X_val_unscaled = scaler.inverse_transform(X_val) 
    X_train_unscaled = scaler.inverse_transform(X_train)
    
    #encoded_imgs = encoder.predict(X_test_unscaled)
    predicted_val = autoencoder.predict(X_val)                    
    predicted_val_unscaled = scaler.inverse_transform(predicted_val)        #unscale
    
    
    
    
    mse_geom = []                       #error of a single geometry
    for k in range(len(X_test)):
        mse_geom.append(sum((predicted_val_unscaled[k,:]-X_val_unscaled[k,:])**2))
    
    mse_val = sum(mse_geom)/len(X_val)        #global mse
    nmse_val = (mse_val/var_orig)             #global nmse
    
    
    
    return(track, dataset, X_train, 
           X_val, X_test, X_train_unscaled, X_val_unscaled, predicted_val, predicted_val_unscaled, 
           mse_val, nmse_val, var_orig, mean_shape)

