# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 17:17:40 2017

@author: Mohit
"""

#Part 1 : Building a CNN

#import Keras packages
import tensorflow as tf
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import load_img
import random

import numpy as np
from keras import applications
from keras.layers import Input
from keras.models import Model
from keras import optimizers
from keras.utils import get_file

img_width, img_height = 256, 256        # Resolution of inputs
train_data_dir = "train"           # Folder of train samples
validation_data_dir = "val" # Folder of validation samples
nb_train_samples = 10000                # Number of train samples
nb_validation_samples = 9500            # Number of validation samples
batch_size = 64                        # Batch size
epochs = 20                # Maximum number of epochs
# Load ResNet50

nb_train_samples = 2543                # Number of train samples
nb_validation_samples = 2261            # Number of validation samples

# Initializing the CNN

#Part 2 - fitting the data set

import glob
Controlled = glob.glob('./normal/*.*')
Diseased = glob.glob('./benign/*.*')
Diseased1 = glob.glob('./malignant/*.*')

data = []
labels = []

for i in Controlled:   
    image=load_img(i, 
    target_size= (256,256))
    image=np.array(image)
    data.append(image)
    labels.append(0)
for i in Diseased:   
    image=load_img(i,  
    target_size= (256,256))
    image=np.array(image)
    data.append(image)
    labels.append(1)

for i in Diseased1:   
    image=load_img(i,  
    target_size= (256,256))
    image=np.array(image)
    data.append(image)
    labels.append(2)

data = np.array(data)
labels = np.array(labels)

from keras.utils.np_utils import to_categorical   

categorical_labels = to_categorical(labels, num_classes=3)


from sklearn.model_selection import train_test_split

#X_train1, X_test0, ytrain1, ytest0 = train_test_split(data, categorical_labels, test_size=0.1,
                                                    #random_state=random.randint(0,100))
X_train1 = data
ytrain1 = categorical_labels

for index in range(1):
    
    classifier = Sequential()

    classifier.add(Convolution2D(32, 3, 3, input_shape = (256, 256, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Convolution2D(16, 3, 3, activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Convolution2D(8, 3, 3, activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))



    classifier.add(Flatten())

    #hidden layer
    classifier.add(Dense(output_dim = 128, activation = 'relu'))
    classifier.add(Dropout(p = 0.5))

    #output layer
    classifier.add(Dense(output_dim = 3, activation = 'softmax'))

    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

    X_train, X_test1, ytrain, ytest1 = train_test_split(X_train1, ytrain1, test_size=0.1,
                                                    random_state=random.randint(0,100))
    
    X_val, X_test, yval, ytest = train_test_split(X_test1, ytest1, test_size=0.5,
                                                    random_state=random.randint(0,100))
    
    from keras.preprocessing.image import ImageDataGenerator
    
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    training_set = train_datagen.flow(
            X_train, ytrain,
            batch_size=64)
    
    val_set = val_datagen.flow(
            X_val, yval,
            batch_size=64)
    
    test_set = test_datagen.flow(
            X_test, ytest,
            batch_size=64)
    
    X, y = test_set.next()
    
    classifier.fit_generator(
            training_set,
            steps_per_epoch=20,
            epochs=100,
            validation_data=val_set,
            validation_steps=100)
    
    w_file = 'BCancer_basic_model_weights_k10.h5'
    classifier.save_weights(w_file)
    
    
    arr = classifier.evaluate(X,y)
    print(arr)
    arr = classifier.predict(X)
    arr = np.argmax(arr, axis=1)

    print(arr)
    print(y)
