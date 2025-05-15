# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 12:35:01 2021

@author: Administrator
"""

import numpy as np
import random
import cv2
import os
from imutils import paths
from keras import applications
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import sys

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Activation
from keras.layers import Flatten, Reshape
from keras.layers import Dense
from keras.optimizers import SGD
from keras import backend as K
import glob
from keras import models
from keras.models import Model
from keras import optimizers
from sklearn.preprocessing import LabelBinarizer
from keras.datasets import cifar10
#from fl_mnist_implementation_tutorial_utils import *
classes = 0

from keras.layers import Input
from keras import layers
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras.preprocessing import image
import keras.backend as K
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs


WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def ResNet50(include_top=True, weights='imagenet',
             input_tensor=None, input_shape=None,
             pooling=None,
             classes=1000):
    """Instantiates the ResNet50 architecture.
    Optionally loads weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format="channels_last"` in your Keras config
    at ~/.keras/keras.json.
    The model and the weights are compatible with both
    TensorFlow and Theano. The data format
    convention used by the model is the one
    specified in your Keras config file.
    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 244)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 197.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=48,
                                      data_format=K.image_data_format(),
					require_flatten=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    #x = ZeroPadding2D((78, 78))(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(img_input)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    if include_top:
        x = Flatten()(x)
        x = Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='resnet50')

    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH,
                                    cache_subdir='models',
                                    md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
        else:
            weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models',
                                    md5_hash='a268eb855778b3df3c7506639542a6af')
        model.load_weights(weights_path)
        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)

        if K.image_data_format() == 'channels_first':
            if include_top:
                maxpool = model.get_layer(name='avg_pool')
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name='fc1000')
                layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')

            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
    return model

def load(paths, verbose=-1):
    '''expects images for each class in seperate dir, 
    e.g all digits in 0 class in the directory named 0 '''
    data = list()
    labels = list()
    # loop over the input images
    
    for (i, imgpath) in enumerate(paths):
        # load the image and extract the class labels
        im_gray = cv2.imread(imgpath)
        im_gray = cv2.resize(im_gray, (256, 256))
        #image = np.array(im_gray).flatten()
        label = imgpath.split(os.path.sep)[-2]
        # scale the image to [0, 1] and add to list
        data.append(im_gray)
        labels.append(label)
        # show an update every `verbose` images
        if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
            print("[INFO] processed {}/{}".format(i + 1, len(paths)))
    # return a tuple of the data and labels
    
    #onehot_encoder = OneHotEncoder(sparse=False)
    
    onehot_encoder = LabelBinarizer()

    labels = np.array(labels)
    labels = labels.reshape(len(labels), 1)
    
    onehot_encoded = onehot_encoder.fit_transform(labels)


    return data, onehot_encoded

#declear path to your mnist data folder
'''
img_path = sys.argv[1]

for path1 in glob.glob(img_path + "/*"):
    classes += 1
#get the path list using the path object
image_paths = list(paths.list_images(img_path))

#apply our function
image_list, label_list = load(image_paths, verbose=10000)

#binarize the labels
lb = LabelBinarizer()
label_list = lb.fit_transform(label_list)

#split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(image_list, 
                                                    label_list, 
                                                    test_size=0.1, 
                                                    random_state=42)



img_path = sys.argv[1]
img_path1 = sys.argv[2]

img_path1 = ""

for path0 in glob.glob(img_path + "/*"):
    img_path1 = path0
    print(img_path1)
    for path1 in glob.glob(img_path1 + "/*"):
        classes += 1
        
'''

classes = 2

import glob
from keras.preprocessing.image import load_img
import random

Controlled = glob.glob('dataset/normal/*.*')
Diseased = glob.glob('dataset/pcos/*.*')

data = []
labels = []

for i in Controlled: 
    print(i)
    image1=load_img(i, 
    target_size= (256,256))
    image1=np.array(image1)
    data.append(image1)
    labels.append(0)
for i in Diseased:   
    image1=load_img(i,  
    target_size= (256,256))
    image1=np.array(image1)
    data.append(image1)
    labels.append(1)

data = np.array(data)
labels = np.array(labels)

from keras.utils.np_utils import to_categorical   

categorical_labels = to_categorical(labels, num_classes=2)


#X_train1, X_test0, ytrain1, ytest0 = train_test_split(data, categorical_labels, test_size=0.1,
                                                    #random_state=random.randint(0,100))
#X_train = data
#y_train = categorical_labels

X_train, X_test, y_train, y_test = train_test_split(data, 
                                                    categorical_labels, 
                                                    test_size=0.1, 
                                                    random_state=random.randint(1,100))

X_val, X_test, y_val, y_test = train_test_split(X_test, 
                                                    y_test, 
                                                    test_size=0.5, 
                                                    random_state=random.randint(1,100))

X_test1 = X_test/255.
y_test1 = y_test

def create_clients(image_list, label_list, num_clients=10, initial='clients'):
    ''' return: a dictionary with keys clients' names and value as 
                data shards - tuple of images and label lists.
        args: 
            image_list: a list of numpy arrays of training images
            label_list:a list of binarized labels for each image
            num_client: number of fedrated members (clients)
            initials: the clients'name prefix, e.g, clients_1 
            
    '''

    #create a list of client names
    client_names = ['{}_{}'.format(initial, i+1) for i in range(num_clients)]

    #randomize the data
    '''
    data = list(zip(image_list, label_list))
    random.shuffle(data)

    #shard data and place at each client
    size = len(data)//num_clients
    shards = [data[i:i + size] for i in range(0, size*num_clients, size)]

    #number of clients must equal number of shards
    assert(len(shards) == len(client_names))

    return {client_names[i] : shards[i] for i in range(len(client_names))} 
    '''
    return client_names

#create clients
clients = create_clients(X_train, y_train, num_clients=10, initial='client')

def batch_data(data_shard, bs=32):
    '''Takes in a clients data shard and create a tfds object off it
    args:
        shard: a data, label constituting a client's data shard
        bs:batch size
    return:
        tfds object'''
    #seperate shard into data and labels lists
    data, label = zip(*data_shard)
    dataset = tf.contrib.data.Dataset.from_tensor_slices((list(data), list(label)))
    return dataset.shuffle(len(label)).batch(bs)

#process and batch the training data for each client
    
#process and batch the test set  
#test_batched = tf.contrib.data.Dataset.from_tensor_slices((X_test, y_test)).batch(len(y_test))
test_batched = list(zip(X_test, y_test))
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras import applications
from keras.layers.normalization import BatchNormalization

img_width, img_height = 256, 256        # Resolution of inputs


class SimpleMLP:
    @staticmethod
    def build(shape, classes1, wts):
        img_width, img_height = 256, 256        # Resolution of inputs

        # Load INCEPTIONV3

        model=ResNet50(include_top=False, weights=None,
                     input_tensor=None, input_shape=(256, 256, 3),
                     pooling=None,
                     classes=1000)
        
        
        #initialise top model
        """
        top_model = Sequential()
        top_model.add(Flatten(input_shape=vgg_model.output_shape[1:]))
        top_model.add(Dense(256, activation='relu'))
        top_model.add(Dropout(0.5))
        top_model.add(Dense(10, activation='softmax'))
        
        WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
        weights_path = get_file('vgg16_weights.h5', WEIGHTS_PATH_NO_TOP)
        
        vgg_model.load_weights(weights_path)
        
        # add the model on top of the convolutional base
        
        model_final = Model(input= vgg_model.input, output= top_model(vgg_model.output))
        
        # Freeze first 15 layers
        for layer in model.layers[:174]:
        	layer.trainable = False
        for layer in model.layers[174:]:
           layer.trainable = True
        """
        
        x = model.output
        x = Flatten()(x)
        x = Dense(1024, activation="relu")(x)
        x = Dropout(0.5)(x)
        x = Dense(1024, activation="relu")(x)
        x = Dropout(0.5)(x)
        predictions = Dense(output_dim = 2, activation="softmax")(x) # 4-way softmax classifier at the end
        
        model_final = Model(input=model.input, output=predictions)
        
        return model_final

        """
        model = Sequential()
        model.add(Reshape((128,128,3), input_shape=(128*128*3,)))
        model.add(Convolution2D(64, 3, 3)) 
        convout1 = Activation('relu')
        model.add(convout1)
        convout2 = MaxPooling2D()
        model.add(convout2)
        
        model.add(Convolution2D(32, 3, 3)) 
        convout1 = Activation('relu')
        model.add(convout1)
        convout2 = MaxPooling2D()
        model.add(convout2)
        
        model.add(Convolution2D(16, 3, 3)) 
        convout1 = Activation('relu')
        model.add(convout1)
        convout2 = MaxPooling2D()
        model.add(convout2)
        
        model.add(Flatten())
        
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(2))
        model.add(Activation('softmax'))
        
        #model.summary()
        """
        #return model_final
    
lr = 0.01 
comms_round = 20
loss='categorical_crossentropy'
metrics = ['accuracy']
optimizer = SGD(lr=lr, 
                decay=lr / comms_round, 
                momentum=0.9
               )  

def weight_scalling_factor(clients_trn_data, client_name):
    client_names = list(clients_trn_data.keys())
    #get the bs
    bs = np.array(list(clients_trn_data[client_name])).shape[0]
    #first calculate the total training data points across clinets
    global_count = sum([tf.compat.v1.data.experimental.cardinality(clients_trn_data[client_name]).numpy() for client_name in client_names])*bs
    # get the total number of data points held by a client
    local_count = tf.compat.v1.data.experimental.cardinality(clients_trn_data[client_name]).numpy()*bs
    return local_count/global_count


def scale_model_weights(weight, scalar):
    '''function for scaling a models weights'''
    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(scalar * weight[i])
    return weight_final



def sum_scaled_weights(scaled_weight_list):
    '''Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights'''
    avg_grad = list()
    #get the average grad accross all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean  = np.sum(grad_list_tuple, axis=0)
        #layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)
        
    return avg_grad


def test_model(X_test1, y_test1,  model, comm_round):
    return model.evaluate(X_test1, y_test1)

def test_model1(X_test, Y_test,  model, comm_round):
    cce = keras.losses.CategoricalCrossentropy(from_logits=True)
    #logits = model.predict(X_test, batch_size=100)
    logits = model.predict(X_test)
    loss = cce(Y_test, logits)
    acc = accuracy_score(tf.argmax(logits, axis=1), tf.argmax(Y_test, axis=1))
    print('comm_round: {} | global_acc: {:.3%} | global_loss: {}'.format(comm_round, acc, loss))
    return acc

#initialize global model
smlp_global = SimpleMLP()
global_model = smlp_global.build(784, 2, "imagenet")
 
global_model.compile(loss=loss, 
                      optimizer=optimizer, 
                      metrics=metrics)
global_model.summary()

local_models = []
data1 = []
labels1 = []
X1, data_1, y1, labels_1 = train_test_split(X_train, 
                                                    y_train, 
                                                    test_size=0.1, 
                                                    random_state=random.randint(1,100))
data1.append(data_1)
labels1.append(labels_1)

X1, data_1, y1, labels_1 = train_test_split(X1, 
                                                    y1, 
                                                    test_size=0.1*10.0/9.0, 
                                                    random_state=random.randint(1,100))

data1.append(data_1)
labels1.append(labels_1)


X1, data_1, y1, labels_1 = train_test_split(X1, 
                                                    y1, 
                                                    test_size=0.1*10.0/8.0, 
                                                    random_state=random.randint(1,100))

data1.append(data_1)
labels1.append(labels_1)

X1, data_1, y1, labels_1 = train_test_split(X1, 
                                                    y1, 
                                                    test_size=0.1*10.0/7.0, 
                                                    random_state=random.randint(1,100))

data1.append(data_1)
labels1.append(labels_1)


X1, data_1, y1, labels_1 = train_test_split(X1, 
                                                    y1, 
                                                    test_size=0.1*10.0/6.0, 
                                                    random_state=random.randint(1,100))

data1.append(data_1)
labels1.append(labels_1)

X1, data_1, y1, labels_1 = train_test_split(X1, 
                                                    y1, 
                                                    test_size=0.1*10.0/5.0, 
                                                    random_state=random.randint(1,100))

data1.append(data_1)
labels1.append(labels_1)

X1, data_1, y1, labels_1 = train_test_split(X1, 
                                                    y1, 
                                                    test_size=0.1*10.0/4.0, 
                                                    random_state=random.randint(1,100))

data1.append(data_1)
labels1.append(labels_1)

X1, data_1, y1, labels_1 = train_test_split(X1, 
                                                    y1, 
                                                    test_size=0.1*10.0/3.0, 
                                                    random_state=random.randint(1,100))

data1.append(data_1)
labels1.append(labels_1)

X1, data_1, y1, labels_1 = train_test_split(X1, 
                                                    y1, 
                                                    test_size=0.1*10.0/2.0, 
                                                    random_state=random.randint(1,100))

data1.append(data_1)
labels1.append(labels_1)

data1.append(X1)
labels1.append(y1)
from keras.preprocessing.image import ImageDataGenerator
 
for i in range(0,1):
    smlp_local = SimpleMLP()
    local_model = smlp_local.build(784, 2, None)
    local_model.compile(loss=loss, 
                  optimizer=optimizer, 
                  metrics=metrics)
    local_models.append(local_model)
#commence global training loop
    
unscaled_weights = list()
client_names= list(clients)


 
class Particle:
    def __init__(self,x0):
        self.position_i=[]          # particle position
        self.velocity_i=[]          # particle velocity
        self.pos_best_i=[]          # best position individual
        self.err_best_i=1          # best error individual
        self.err_i=1               # error individual

        for i in range(0,num_dimensions):
            #print(num_dimensions)
            self.velocity_i.append(random.uniform(0,1))
            self.position_i.append(x0[i])

    # evaluate current fitness
    def evaluate(self,costFunc):
        self.err_i=costFunc(self.position_i)

        # check to see if the current position is an individual best
        if self.err_i < self.err_best_i or self.err_best_i==1:
            self.pos_best_i=self.position_i
            self.err_best_i=self.err_i
        #print(self.err_i, self.err_best_i, self.pos_best_i)

    # update new particle velocity
    def update_velocity(self,pos_best_g):
        w=0.5       # constant inertia weight (how much to weigh the previous velocity)
        c1=1        # cognative constant
        c2=2        # social constant


        for i in range(0,num_dimensions):
            r1=random.random()
            r2=random.random()

            vel_cognitive=c1*r1*(self.pos_best_i[i]-self.position_i[i])
            vel_social=c2*r2*(pos_best_g[i]-self.position_i[i])
            self.velocity_i[i]=w*self.velocity_i[i]+vel_cognitive+vel_social

    # update the particle position based off new velocity updates
    def update_position(self,bounds):
        for i in range(0,num_dimensions):
            self.position_i[i]=self.position_i[i]+self.velocity_i[i]

            # adjust maximum position if necessary
            if self.position_i[i]>bounds[i][1]:
                self.position_i[i]=bounds[i][1]

            # adjust minimum position if neseccary
            if self.position_i[i] < bounds[i][0]:
                self.position_i[i]=bounds[i][0]




    
for comm_round in range(comms_round):
     
    '''       
    clients_batched = dict()
    for (client_name, data) in clients.items():
        clients_batched[client_name] = zip(*data)

    '''
    client_names= list(clients)
    random.shuffle(client_names)

    indv = []
    bounds = []
    #weights1_shape = np.array(weights_arr[0][idx]).shape
    
    for k in range(0, len(client_names)):
        indv.append(random.uniform(0,1))
        #indv.append(5)
        bounds.append((0,1))


    # get the global model's weights - will serve as the initial weights for all local models
    global_weights = global_model.get_weights()
    
    #initial list to collect local model weights after scalling
    unscaled_weights = list()
    #randomize client data - using keys
    
    #loop through each client and create new local model
    idx = 0
    for client in client_names:
        '''
        smlp_local = SimpleMLP()
        local_model = smlp_local.build(784, 2, None)
        local_model.compile(loss=loss, 
                      optimizer=optimizer, 
                      metrics=metrics)
        '''
        print('client', idx)
        local_model = local_models[0]
        
        #set local model weight to the weight of the global model
        local_model.set_weights(global_weights)
        
        #fit local model with client's data
        #print(clients_batched[client])
        '''
        data1 = tuple(clients_batched[client])
        data_x = np.array(data1[0])
        data_y = np.array(data1[1])
        '''
        data_x = data1[idx]
        data_y = labels1[idx]
        
        train_datagen = ImageDataGenerator(
            rescale=1./255.,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
    
        test_datagen = ImageDataGenerator(rescale=1./255.)
        
        val_datagen = ImageDataGenerator(rescale=1./255.)
        
        training_set = train_datagen.flow(
                np.array(data1[idx]), np.array(labels1[idx]),
                batch_size=64)
        
        val_set = val_datagen.flow(
                np.array(X_val), np.array(y_val),
                batch_size=64)
        
  
        idx += 1

        
        #local_model.fit(data_x, data_y, batch_size=4, epochs=1, steps_per_epoch=20, verbose=1)
        local_model.fit_generator(
            training_set,
            steps_per_epoch=20,
            epochs=2,
            validation_data=val_set,
            validation_steps=50)

        unscaled_weights.append(local_model.get_weights())    

    '''
    def func1(x):

        idx = 0
        scaled_local_weight_list = list()

        for client in client_names:
        
    
            #scale the model weights and add to list
            #scaling_factor = weight_scalling_factor(clients_batched, client)
            scaled_weights = scale_model_weights(unscaled_weights[idx], x[idx]/np.sum(x))
            scaled_local_weight_list.append(scaled_weights)
            idx += 1
            
            #clear session to free memory after each communication round
            #K.clear_session()
            #break
        
        #to get the average over all the local model, we simply take the sum of the scaled weights
        average_weights = sum_scaled_weights(scaled_local_weight_list)
        
        #update global model 
        global_model.set_weights(average_weights)
        #print(np.array(X_test1).shape, np.array(y_test1).shape)
        ret = global_model.evaluate(np.array(X_test1), np.array(y_test1), batch_size=32)
        #print('Iteration: ', comm_round, ret)
        print(x, ret)
        return 1-ret[1]    
    
    def PSO(costFunc,x0,bounds,num_particles,maxiter):
    
    
        global num_dimensions
        num_dimensions=len(x0)
        err_best_g=1                   # best error for group
        pos_best_g=[]                   # best position for group
    
        # establish the swarm
        swarm=[]
        for i in range(0,num_particles):
            swarm.append(Particle(x0))
    
        # begin optimization loop
        i=0
        prev = 0
        repeat = 0
        while i < maxiter:
            #print i,err_best_g
            # cycle through particles in swarm and evaluate fitness
            for j in range(0,num_particles):
                swarm[j].evaluate(costFunc)
    
                # determine if current particle is the best (globally)
                if swarm[j].err_i < err_best_g or err_best_g == 1:
                    pos_best_g=list(swarm[j].position_i)
                    err_best_g=float(swarm[j].err_i)
    
            # cycle through swarm and update velocities and position
            for j in range(0,num_particles):
                swarm[j].update_velocity(pos_best_g)
                swarm[j].update_position(bounds)
            i+=1
            print('Iteration', i, pos_best_g)
            #print(func1(pos_best_g), pos_best_g)
            
            """
            if(prev == func1(pos_best_g)):
                repeat += 1
                if(repeat >= 15):
                    break
            else:
                repeat = 0
            prev = func1(pos_best_g)
            """
        return pos_best_g
    
    
    par1 = PSO(func1,indv,bounds,num_particles=5,maxiter=5)
    '''
    
    idx = 0
    scaled_local_weight_list = list()
    for client in client_names:
    

        #scale the model weights and add to list
        #scaling_factor = weight_scalling_factor(clients_batched, client)
        scaled_weights = scale_model_weights(unscaled_weights[idx], 0.1)
        scaled_local_weight_list.append(scaled_weights)
        idx += 1
        
        #clear session to free memory after each communication round
        #K.clear_session()
        #break
    
    #to get the average over all the local model, we simply take the sum of the scaled weights
    average_weights = sum_scaled_weights(scaled_local_weight_list)
    
    #update global model 
    global_model.set_weights(average_weights)
    #print(np.array(X_test1).shape, np.array(y_test1).shape)
    ret = global_model.evaluate(np.array(X_test1), np.array(y_test1), batch_size=32)
    print('Comms round: ', comm_round, ret)
    """
    #test global model and print out metrics after each communications round
    for i in test_batched:
        data1 = tuple(zip(*i))
        X_test = np.array(data1[0])
        y_test = np.array(data1[1])
        for idx in range(200):
            global_acc, global_loss = test_model(X_test[idx], y_test[idx], global_model, comm_round)
            
            print(global_acc, global_loss)
    """        
print(np.array(X_test1).shape, np.array(y_test1).shape)
print(global_model.evaluate(np.array(X_test1), np.array(y_test1), batch_size=32))
global_model.save_weights('Federated.h5')
       