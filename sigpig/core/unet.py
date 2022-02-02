#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
U-Net models for Rattlesnake Ridge data processing.
"""

import tensorflow as tf

def make_large_unet(fac, sr):
    # BUILD THE MODEL
    # These models start with an input
    if sr==40:
        input_layer=tf.keras.layers.Input(shape=(600,2)) # 1 Channel seismic data
      
        #These Convolutional blocks expect 2D data (time-steps x channels)
        #This is just one channel, but if you wanted to add more stations as extra channels you can
        #network=tf.keras.layers.Reshape((600,2))(input_layer)
        
        # build the network, here is your first convolution layer
        level1=tf.keras.layers.Conv1D(int(fac*32),21,activation='relu',padding='same')(input_layer) # N filters, Filter Size, Stride, padding
        
        # This layer is a trick of the trade it helps training deeper networks, by keeping gradients close to the same scale
        #network=tf.keras.layers.BatchNormalization()(level1)
        
        #Max Pooling Layer
        network=tf.keras.layers.MaxPooling1D()(level1) #300
        
        #Next Block
        level2=tf.keras.layers.Conv1D(int(fac*64),15,activation='relu',padding='same')(network)
        #network=tf.keras.layers.BatchNormalization()(level2)
        network=tf.keras.layers.MaxPooling1D()(level2) #150
        
        #Next Block
        level3=tf.keras.layers.Conv1D(int(fac*128),11,activation='relu',padding='same')(network)
        #network=tf.keras.layers.BatchNormalization()(level3)
        network=tf.keras.layers.MaxPooling1D()(level3) #75
        
        #Base of Network
        network=tf.keras.layers.Flatten()(network)
        base_level=tf.keras.layers.Dense(75,activation='relu')(network)
        
        #network=tf.keras.layers.BatchNormalization()(base_level)
        network=tf.keras.layers.Reshape((75,1))(base_level)
        
        #Upsample and add skip connections
        network=tf.keras.layers.Conv1D(int(fac*128),11,activation='relu',padding='same')(network) # N filters, Filter Size, Stride, padding
        network=tf.keras.layers.UpSampling1D()(network)
        level3=tf.keras.layers.Concatenate()([network,level3]) # N filters, Filter Size, Stride, padding
        
        #Upsample and add skip connections
        network=tf.keras.layers.Conv1D(int(fac*64),15,activation='relu',padding='same')(level3) # N filters, Filter Size, Stride, padding
        network=tf.keras.layers.UpSampling1D()(network)
        level2=tf.keras.layers.Concatenate()([network,level2]) # N filters, Filter Size, Stride, padding
        
        #Upsample and add skip connections
        network=tf.keras.layers.Conv1D(int(fac*32),21,activation='relu',padding='same')(level2) # N filters, Filter Size, Stride, padding
        network=tf.keras.layers.UpSampling1D()(network)
        level1=tf.keras.layers.Concatenate()([network,level1]) # N filters, Filter Size, Stride, padding
        
        #End of network
        network=tf.keras.layers.Conv1D(1,21,activation='sigmoid',padding='same')(level1) # N filters, Filter Size, Stride, padding
        output=tf.keras.layers.Flatten()(network) # N filters, Filter Size, Stride, padding
        
        model=tf.keras.models.Model(input_layer,output)
        opt = tf.keras.optimizers.Adam(lr=0.0001)
        model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])
        
    elif sr==100:
        input_layer=tf.keras.layers.Input(shape=(1500,2)) # 1 Channel seismic data
        
        # First block
        level1=tf.keras.layers.Conv1D(int(fac*32),21,activation='relu',padding='same')(input_layer) # N filters, Filter Size, Stride, padding
        network=tf.keras.layers.MaxPooling1D()(level1) #750
        
        # Second Block
        level2=tf.keras.layers.Conv1D(int(fac*64),15,activation='relu',padding='same')(network)
        network=tf.keras.layers.MaxPooling1D()(level2) #375
        network=tf.keras.layers.ZeroPadding1D((0,1))(network)
        
        #Next Block
        level3=tf.keras.layers.Conv1D(int(fac*128),11,activation='relu',padding='same')(network)
        #network=tf.keras.layers.BatchNormalization()(level3)
        network=tf.keras.layers.MaxPooling1D()(level3) #188
        
        #Base of Network
        network=tf.keras.layers.Flatten()(network)
        base_level=tf.keras.layers.Dense(188,activation='relu')(network)
        
        #network=tf.keras.layers.BatchNormalization()(base_level)
        network=tf.keras.layers.Reshape((188,1))(base_level)
        
        #Upsample and add skip connections
        network=tf.keras.layers.Conv1D(int(fac*128),11,activation='relu',padding='same')(network) # N filters, Filter Size, Stride, padding
        network=tf.keras.layers.UpSampling1D()(network)
        
        level3=tf.keras.layers.Concatenate()([network,level3]) # N filters, Filter Size, Stride, padding
        level3=tf.keras.layers.Lambda( lambda x: x[:,:-1,:])(level3)
        
        #Upsample and add skip connections
        network=tf.keras.layers.Conv1D(int(fac*64),15,activation='relu',padding='same')(level3) # N filters, Filter Size, Stride, padding
        network=tf.keras.layers.UpSampling1D()(network)
        level2=tf.keras.layers.Concatenate()([network,level2]) # N filters, Filter Size, Stride, padding
        
        #Upsample and add skip connections
        network=tf.keras.layers.Conv1D(int(fac*32),21,activation='relu',padding='same')(level2) # N filters, Filter Size, Stride, padding
        network=tf.keras.layers.UpSampling1D()(network)
        level1=tf.keras.layers.Concatenate()([network,level1]) # N filters, Filter Size, Stride, padding
        
        #End of network
        network=tf.keras.layers.Conv1D(1,21,activation='sigmoid',padding='same')(level1) # N filters, Filter Size, Stride, padding
        output=tf.keras.layers.Flatten()(network) # N filters, Filter Size, Stride, padding
        
        model=tf.keras.models.Model(input_layer,output)
        opt = tf.keras.optimizers.Adam(lr=0.0001)
        model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])
        
    if sr==250:
        input_layer=tf.keras.layers.Input(shape=(120,2)) # 1 Channel seismic data
      
        #These Convolutional blocks expect 2D data (time-steps x channels)
        #This is just one channel, but if you wanted to add more stations as extra channels you can
        #network=tf.keras.layers.Reshape((600,2))(input_layer)
        
        # build the network, here is your first convolution layer
        level1=tf.keras.layers.Conv1D(int(fac*32),21,activation='relu',padding='same')(input_layer) # N filters, Filter Size, Stride, padding
        
        # This layer is a trick of the trade it helps training deeper networks, by keeping gradients close to the same scale
        #network=tf.keras.layers.BatchNormalization()(level1)
        
        #Max Pooling Layer
        network=tf.keras.layers.MaxPooling1D()(level1) #60
        
        #Next Block
        level2=tf.keras.layers.Conv1D(int(fac*64),15,activation='relu',padding='same')(network)
        #network=tf.keras.layers.BatchNormalization()(level2)
        network=tf.keras.layers.MaxPooling1D()(level2) #30
        
        #Next Block
        level3=tf.keras.layers.Conv1D(int(fac*128),11,activation='relu',padding='same')(network)
        #network=tf.keras.layers.BatchNormalization()(level3)
        network=tf.keras.layers.MaxPooling1D()(level3) #15
        
        #Base of Network
        network=tf.keras.layers.Flatten()(network)
        base_level=tf.keras.layers.Dense(15,activation='relu')(network)
        
        #network=tf.keras.layers.BatchNormalization()(base_level)
        network=tf.keras.layers.Reshape((15,1))(base_level)
        
        #Upsample and add skip connections
        network=tf.keras.layers.Conv1D(int(fac*128),11,activation='relu',padding='same')(network) # N filters, Filter Size, Stride, padding
        network=tf.keras.layers.UpSampling1D()(network)
        level3=tf.keras.layers.Concatenate()([network,level3]) # N filters, Filter Size, Stride, padding
        
        #Upsample and add skip connections
        network=tf.keras.layers.Conv1D(int(fac*64),15,activation='relu',padding='same')(level3) # N filters, Filter Size, Stride, padding
        network=tf.keras.layers.UpSampling1D()(network)
        level2=tf.keras.layers.Concatenate()([network,level2]) # N filters, Filter Size, Stride, padding
        
        #Upsample and add skip connections
        network=tf.keras.layers.Conv1D(int(fac*32),21,activation='relu',padding='same')(level2) # N filters, Filter Size, Stride, padding
        network=tf.keras.layers.UpSampling1D()(network)
        level1=tf.keras.layers.Concatenate()([network,level1]) # N filters, Filter Size, Stride, padding
        
        #End of network
        network=tf.keras.layers.Conv1D(1,21,activation='sigmoid',padding='same')(level1) # N filters, Filter Size, Stride, padding
        output=tf.keras.layers.Flatten()(network) # N filters, Filter Size, Stride, padding
        
        model=tf.keras.models.Model(input_layer,output)
        opt = tf.keras.optimizers.Adam(lr=0.0001)
        model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])
    
    model.summary()
    
    return model

def make_large_unet_drop(fac,sr):
    
    if sr==40:
        # These models start with an input
        input_layer=tf.keras.layers.Input(shape=(600,2)) # 1 Channel seismic data
        
        #These Convolutional blocks expect 2D data (time-steps x channels)
        #This is just one channel, but if you wanted to add more stations as extra channels you can
        #network=tf.keras.layers.Reshape((600,2))(input_layer)
        
        # build the network, here is your first convolution layer
        level1=tf.keras.layers.Conv1D(int(fac*32),21,activation='relu',padding='same')(input_layer) # N filters, Filter Size, Stride, padding
        
        # This layer is a trick of the trade it helps training deeper networks, by keeping gradients close to the same scale
        #network=tf.keras.layers.BatchNormalization()(level1)
        
        #Max Pooling Layer
        network=tf.keras.layers.MaxPooling1D()(level1) #300
        
        #Next Block
        level2=tf.keras.layers.Conv1D(int(fac*64),15,activation='relu',padding='same')(network)
        #network=tf.keras.layers.BatchNormalization()(level2)
        network=tf.keras.layers.MaxPooling1D()(level2) #150
        
        #Next Block
        level3=tf.keras.layers.Conv1D(int(fac*128),11,activation='relu',padding='same')(network)
        #network=tf.keras.layers.BatchNormalization()(level3)
        network=tf.keras.layers.MaxPooling1D()(level3) #75
        
        #Base of Network
        network=tf.keras.layers.Flatten()(network)
        base_level=tf.keras.layers.Dense(75,activation='relu')(network)
        
        #network=tf.keras.layers.BatchNormalization()(base_level)
        network=tf.keras.layers.Reshape((75,1))(base_level)
        
        #Upsample and add skip connections
        network=tf.keras.layers.Conv1D(int(fac*128),11,activation='relu',padding='same')(network) # N filters, Filter Size, Stride, padding
        network=tf.keras.layers.UpSampling1D()(network)
        level3=tf.keras.layers.Concatenate()([network,level3]) # N filters, Filter Size, Stride, padding
        
        # Upsample and add skip connections
        network=tf.keras.layers.Conv1D(int(fac*64),15,activation='relu',padding='same')(level3) # N filters, Filter Size, Stride, padding
        network=tf.keras.layers.UpSampling1D()(network)
        level2=tf.keras.layers.Concatenate()([network,level2]) # N filters, Filter Size, Stride, padding
        
        # Upsample and add skip connections
        network=tf.keras.layers.Conv1D(int(fac*32),21,activation='relu',padding='same')(level2) # N filters, Filter Size, Stride, padding
        network=tf.keras.layers.UpSampling1D()(network)
        level1=tf.keras.layers.Concatenate()([network,level1]) # N filters, Filter Size, Stride, padding
        
        # End of network
        network=tf.keras.layers.Dropout(.2)(level1)
        network=tf.keras.layers.Conv1D(1,21,activation='sigmoid',padding='same')(network) # N filters, Filter Size, Stride, padding
        output=tf.keras.layers.Flatten()(network) # N filters, Filter Size, Stride, padding
        
        model=tf.keras.models.Model(input_layer,output)
        opt = tf.keras.optimizers.Adam(lr=0.0001)
        model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])
    
    elif sr==100:
        input_layer=tf.keras.layers.Input(shape=(1500,2)) # 1 Channel seismic data
        
        # First block
        level1=tf.keras.layers.Conv1D(int(fac*32),21,activation='relu',padding='same')(input_layer) # N filters, Filter Size, Stride, padding
        network=tf.keras.layers.MaxPooling1D()(level1) #750
        
        # Second Block
        level2=tf.keras.layers.Conv1D(int(fac*64),15,activation='relu',padding='same')(network)
        network=tf.keras.layers.MaxPooling1D()(level2) #375
        network=tf.keras.layers.ZeroPadding1D((0,1))(network)
        
        #Next Block
        level3=tf.keras.layers.Conv1D(int(fac*128),11,activation='relu',padding='same')(network)
        #network=tf.keras.layers.BatchNormalization()(level3)
        network=tf.keras.layers.MaxPooling1D()(level3) #188
        
        #Base of Network
        network=tf.keras.layers.Flatten()(network)
        base_level=tf.keras.layers.Dense(188,activation='relu')(network)
        
        #network=tf.keras.layers.BatchNormalization()(base_level)
        network=tf.keras.layers.Reshape((188,1))(base_level)
        
        #Upsample and add skip connections
        network=tf.keras.layers.Conv1D(int(fac*128),11,activation='relu',padding='same')(network) # N filters, Filter Size, Stride, padding
        network=tf.keras.layers.UpSampling1D()(network)
        
        level3=tf.keras.layers.Concatenate()([network,level3]) # N filters, Filter Size, Stride, padding
        level3=tf.keras.layers.Lambda( lambda x: x[:,:-1,:])(level3)
        
        #Upsample and add skip connections
        network=tf.keras.layers.Conv1D(int(fac*64),15,activation='relu',padding='same')(level3) # N filters, Filter Size, Stride, padding
        network=tf.keras.layers.UpSampling1D()(network)
        level2=tf.keras.layers.Concatenate()([network,level2]) # N filters, Filter Size, Stride, padding
        
        #Upsample and add skip connections
        network=tf.keras.layers.Conv1D(int(fac*32),21,activation='relu',padding='same')(level2) # N filters, Filter Size, Stride, padding
        network=tf.keras.layers.UpSampling1D()(network)
        level1=tf.keras.layers.Concatenate()([network,level1]) # N filters, Filter Size, Stride, padding
        
        #End of network
        network=tf.keras.layers.Dropout(.2)(level1)
        network=tf.keras.layers.Conv1D(1,21,activation='sigmoid',padding='same')(level1) # N filters, Filter Size, Stride, padding
        output=tf.keras.layers.Flatten()(network) # N filters, Filter Size, Stride, padding
        
        model=tf.keras.models.Model(input_layer,output)
        opt = tf.keras.optimizers.Adam(lr=0.0001)
        model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])
    
    model.summary()
    
    return model