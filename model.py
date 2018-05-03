
import pandas as pd
import numpy as np
import keras
from keras.layers import Activation, Input, Dense, Embedding, LSTM, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Concatenate, Dropout
from keras.models import Model
from keras.models import load_model
from keras.layers.convolutional import Cropping2D
from keras.layers.core import Lambda
from keras import optimizers
from PIL import Image
import os.path
import os
import random
import socket
import json
import matplotlib.pyplot as plt
import download_dataset_file
import avisala


DATA_DIRECTORY = 'data'
MINIBATCH_SIZE = 64
# split dataset in train (70%), validation (10%) and test (20%)
TRAIN_DATASET_SIZE = 0.7
VALIDATION_DATASET_SIZE = 0.1


# datapoints in 'data/driving_log.csv' file contains filenames and steers

if not os.path.isfile(DATA_DIRECTORY + os.sep + 'driving_log.csv'):
    download_dataset_file.download_dataset_file(DATA_DIRECTORY)

df = pd.read_csv(DATA_DIRECTORY + os.sep + 'driving_log.csv', header = None,
    names = ['file_image_center','file_image_left','file_image_right','steer','acc', 'break', 'speed'])

# number of datapoints in whole dataset
dataset_length = df.shape[0]

#if dataset_length > 256:
#    dataset_length = 256

# auxiliary index to split and shuffle dataset
idx = list(range(dataset_length))
random.shuffle(idx)
number_of_datapoints_in_train_dataset = int(dataset_length * TRAIN_DATASET_SIZE)
number_of_datapoints_in_validation_dataset = int(dataset_length * VALIDATION_DATASET_SIZE)
number_of_datapoints_in_test_dataset = dataset_length - number_of_datapoints_in_train_dataset - number_of_datapoints_in_validation_dataset

# create auxiliary index for each dataset
idx_train = idx[:number_of_datapoints_in_train_dataset]
idx_validation = idx[number_of_datapoints_in_train_dataset:(number_of_datapoints_in_train_dataset+number_of_datapoints_in_validation_dataset)]
idx_test = idx[(number_of_datapoints_in_train_dataset+number_of_datapoints_in_validation_dataset):]

# figure out image size
file_image_center = DATA_DIRECTORY +  os.sep + 'IMG' + os.sep + df.iloc[0]['file_image_center']
image_center = Image.open(file_image_center)
image_center = np.asarray(image_center)
image_shape = image_center.shape

# generetor for train, validation and test datasets
def get_data_generador(df, idx, chunk_size, image_shape, sub_dataset_type):
    idx_base_chunk = 0
    while True:
        this_chunk_size = chunk_size
        if idx_base_chunk + this_chunk_size > len(idx):
            this_chunk_size = len(idx) - idx_base_chunk
        np_image = np.zeros((this_chunk_size, image_shape[0], image_shape[1], image_shape[2]))
        np_steer = np.zeros((this_chunk_size))
        for i in range(this_chunk_size):
            file_image_center = DATA_DIRECTORY + os.sep + 'IMG' + os.sep + df.iloc[idx[i+idx_base_chunk]]['file_image_center']
            steer = df.iloc[idx[i+idx_base_chunk]]['steer']
            image_center = np.array(Image.open(file_image_center).convert("RGB"))
            np_image[i+0,:,:,:] = image_center
            np_steer[i+0] = float(steer)
        idx_base_chunk = idx_base_chunk + this_chunk_size
        if (idx_base_chunk == len(idx)):
            idx_base_chunk = 0
        yield np_image, np_steer

# return keras model folling layers:
#    x
#    normalization
#    conv2d --> relu --> maxpooling
#    conv2d --> relu --> maxpooling
#    conv2d --> relu --> maxpooling --> flatten
#    dense --> relu -->  dropout
#    dense
#    predict_steer
def buildModel(image_shape):
    input_image = Input(shape = image_shape, name = 'image')
    x = input_image
    x = Lambda(lambda x: (x/255.0)-0.5)(x)

    x = Conv2D(24, kernel_size = (10,10), strides  = (1,1), padding = 'same', name = 'conv_1', kernel_initializer = keras.initializers.glorot_uniform())(x)
    x = Activation('relu', name = 'relu_1')(x)
    x = MaxPooling2D((3, 6), strides=(2, 6), name = 'maxpool_1')(x)

    x = Conv2D(36, kernel_size = (7,7), strides  = (1,1), padding = 'same', name = 'conv_2', kernel_initializer = keras.initializers.glorot_uniform())(x)
    x = Activation('relu', name = 'relu_2')(x)
    x = MaxPooling2D((3, 6), strides=(2, 6), name = 'maxpool_2')(x)

    x = Conv2D(48, kernel_size = (5,5), strides  = (1,1), padding = 'same', name = 'conv_3', kernel_initializer = keras.initializers.glorot_uniform())(x)
    x = Activation('relu', name = 'relu_3')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name = 'maxpool_3')(x)

    x = Flatten()(x)
    fc1 = Dense(512, name='fc1', kernel_initializer = keras.initializers.glorot_uniform())(x)
    fc1 = Activation('relu')(fc1)
    fc1 = Dropout(0.5, name = 'dropout_fc1')(fc1)

    fc3 = Dense(16, name='fc3', kernel_initializer = keras.initializers.glorot_uniform())(fc1)

    predict_steer = Dense(1, name='predict_steer', kernel_initializer = keras.initializers.glorot_uniform())(fc3)
    model = Model(inputs = input_image, outputs = predict_steer, name='CarDriver')

    return model

model = None

# if there is CarDriver.h5 file with already trained model, try to continue training
if os.path.isfile('model.h5'):
    model = load_model('model.h5')
# else build a fresh model
else:
    model = buildModel(image_shape)

print ('model', model)
model.summary()

# set stochastic gradient descent as optimizer with learning rate 0.0075, decay 1ee-6, momentum 0.9 and nesterov
sgd = optimizers.SGD(lr=0.0075, decay=1e-6, momentum=0.9, nesterov=True)
# regression problem --> loss is mean_squared_error
model.compile(optimizer=sgd, loss='mean_squared_error')

# save trained model after each epoch
callbacks_list = [
    keras.callbacks.ModelCheckpoint(filepath='model.h5', monitor='val_loss', save_best_only=False)
]

# split indicex dataset
idx_train = idx[:number_of_datapoints_in_train_dataset]
idx_validation = idx[number_of_datapoints_in_train_dataset:(number_of_datapoints_in_train_dataset+number_of_datapoints_in_validation_dataset)]
idx_test = idx[(number_of_datapoints_in_train_dataset+number_of_datapoints_in_validation_dataset):]

# train the model
history = model.fit_generator(
    get_data_generador(df, idx_train, MINIBATCH_SIZE, image_shape, 'train'),
    steps_per_epoch = int(number_of_datapoints_in_train_dataset / MINIBATCH_SIZE) + 1,
    epochs = 120,
    validation_data = get_data_generador(df, idx_validation, MINIBATCH_SIZE, image_shape, 'validation'),
    validation_steps = int(number_of_datapoints_in_validation_dataset / MINIBATCH_SIZE) + 1,
    callbacks = callbacks_list
)

model.save('model.h5')


fig = plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
fig.savefig('train_history.png')

test_loss = model.evaluate_generator(
    get_data_generador(df, idx_test, MINIBATCH_SIZE, image_shape, 'test'),
    int(number_of_datapoints_in_test_dataset / MINIBATCH_SIZE) + 1
)

print ('test_loss', test_loss)
