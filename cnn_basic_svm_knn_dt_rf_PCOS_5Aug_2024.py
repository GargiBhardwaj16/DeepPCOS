# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 17:17:40 2017

@author: Mohit
"""

#Part 1 : Building a CNN

#import Keras packages
#import tensorflow as tf
import numpy as np
#import keras
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
classifier.add(Dense(output_dim = 2, activation = 'softmax'))

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

nb_train_samples = 2543                # Number of train samples
nb_validation_samples = 2261            # Number of validation samples

# Initializing the CNN

#Part 2 - fitting the data set

import glob
Controlled = glob.glob('dataset/normal/*.*')
Diseased = glob.glob('dataset/pcos/*.*')
#Diseased1 = glob.glob('./malignant/*.*')

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
'''
for i in Diseased1:   
    image=load_img(i,  
    target_size= (256,256))
    image=np.array(image)
    data.append(image)
    labels.append(2)
'''
data = np.array(data)
labels = np.array(labels)

from keras.utils.np_utils import to_categorical   

categorical_labels = to_categorical(labels, num_classes=2)

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from keras import backend as K

#X_train1, X_test0, ytrain1, ytest0 = train_test_split(data, categorical_labels, test_size=0.1,
                                                    #random_state=random.randint(0,100))
X_train1 = data
ytrain1 = categorical_labels

for index in range(1):
    
    #classifier = Model(input=model.input, output=predictions)

    #classifier.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=1e-3, momentum=0.9), metrics=["accuracy"])


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
            batch_size=500)
    
    val_set = val_datagen.flow(
            X_val, yval,
            batch_size=64)
    
    test_set = test_datagen.flow(
            X_test, ytest,
            batch_size=64)
    
    X_train, ytrain = training_set.next()
    X_test, ytest = test_set.next()
    
  
    
    w_file = 'Federated_CNN.h5'
    classifier.load_weights(w_file)
    
    
    arr = classifier.evaluate(X_test,ytest) 
    print(arr)

    num = len(classifier.layers)
    print(num)
    #layer_outputs = [classifier.layers[num-1].output] 
    # Extracts the outputs of the top 12 layers
    #activation_model = Model(inputs=model.input, outputs=classifier.layers[num-1].output) # Creates a model that will return these outputs, given the model input

    get_1st_layer_output = K.function([classifier.layers[0].input],
    [classifier.layers[num-2].output])
    print(X_train.shape)
    #X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    #X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    
    layer_output = get_1st_layer_output([X_train])
    layer_output1 = get_1st_layer_output([X_test])
    
    print(layer_output[0].shape)
    
    svm = SVC(kernel='rbf')
    knn = KNeighborsClassifier()
    dt = DecisionTreeClassifier(random_state=9)
    rf = RandomForestClassifier(n_estimators=100, random_state=9)
    y_train = np.argmax(ytrain, axis=1)
    y_test = np.argmax(ytest, axis=1)
    svm.fit(layer_output[0], y_train)
    knn.fit(layer_output[0], y_train)
    dt.fit(layer_output[0], y_train)
    rf.fit(layer_output[0], y_train)
    
    arr1 = svm.predict(layer_output1[0])
    print('SVM: ', arr1, y_test)
    corr = 0
    for index in range(0, len(arr1)):
        #print(index)
        if(arr1[index] == y_test[index]):
            corr += 1
    print('SVM accuracy: ', corr/len(arr1))
    
    arr1 = knn.predict(layer_output1[0])
    print('KNN: ', arr1, y_test)
    corr = 0
    for index in range(0, len(arr1)):
        #print(index)
        if(arr1[index] == y_test[index]):
            corr += 1
    print('k-NN accuracy: ', corr/len(arr1))
    
    arr1 = dt.predict(layer_output1[0])
    print('DT: ', arr1, y_test)
    corr = 0
    for index in range(0, len(arr1)):
        #print(index)
        if(arr1[index] == y_test[index]):
            corr += 1
    print('DT accuracy: ', corr/len(arr1))
    
    arr1 = rf.predict(layer_output1[0])
    print('RF: ', arr1, y_test)
    corr = 0
    for index in range(0, len(arr1)):
        #print(index)
        if(arr1[index] == y_test[index]):
            corr += 1
    print('RF accuracy: ', corr/len(arr1))