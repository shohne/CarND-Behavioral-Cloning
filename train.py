
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
import random
import socket
import json











class Avisala_Callback(keras.callbacks.Callback):
    def get_connection(self):
        host = socket.gethostname()
        port = 20081
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((host, port))
        return s

    def send_dict(self, s, dict):
        str_content = json.dumps(dict)
        content = str_content.encode()
        s.send(content)
        return


    def on_train_begin(self, logs={}):
        self.logs = []
        notification = dict()
        notification['device'] = 'HOHNE_IOS'
        notification['model'] = 'CarDriver'
        notification['event'] = 'on_train_begin'
        notification['train_loss'] = ''
        s = self.get_connection()
        self.send_dict(s, notification)
        s.close()
        return

    def on_train_end(self, logs={}):
        self.logs.append(logs)
        notification = dict()
        notification['device'] = 'HOHNE_IOS'
        notification['model'] = 'CarDriver'
        notification['event'] = 'on_train_end'
        notification['train_loss'] = str(logs.get('loss'))
        s = self.get_connection()
        self.send_dict(s, notification)
        s.close()
        return

    def on_epoch_begin(self, epoch, logs={}):
        self.logs.append(logs)
        notification = dict()
        notification['device'] = 'HOHNE_IOS'
        notification['model'] = 'CarDriver'
        notification['event'] = 'on_epoch_begin'
        notification['train_loss'] = str(logs.get('loss'))
        s = self.get_connection()
        self.send_dict(s, notification)
        s.close()
        return

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        notification = dict()
        notification['device'] = 'HOHNE_IOS'
        notification['model'] = 'CarDriver'
        notification['event'] = 'on_epoch_end'
        notification['train_loss'] = str(logs.get('loss'))
        s = self.get_connection()
        self.send_dict(s, notification)
        s.close()
        return

    def on_batch_begin(self, batch, logs={}):
        if True:
            return
        self.logs.append(logs)
        notification = dict()
        notification['device'] = 'HOHNE_IOS'
        notification['model'] = 'CarDriver'
        notification['event'] = 'on_batch_begin'
        notification['train_loss'] = str(logs.get('loss'))
        s = self.get_connection()
        self.send_dict(s, notification)
        s.close()
        return

    def on_batch_end(self, batch, logs={}):
        if True:
            return
        self.logs.append(logs)
        notification = dict()
        notification['device'] = 'HOHNE_IOS'
        notification['model'] = 'CarDriver'
        notification['event'] = 'on_batch_begin'
        notification['train_loss'] = str(logs.get('loss'))
        s = self.get_connection()
        self.send_dict(s, notification)
        s.close()
        return








#DATA_DIRECTORY = './data/road1/'
DATA_DIRECTORY = './data/road2/'
MINIBATCH_SIZE = 64

df = pd.read_csv(DATA_DIRECTORY + 'driving_log.csv', header = None,
    names = ['file_image_center','file_image_left','file_image_right','steer','acc', 'break', 'speed'])


def get_data(df):
    dataset_length = df.shape[0]
#    if dataset_length > 256:
#        dataset_length = 256
    print ('dataset_length', dataset_length)
    file_image_center = DATA_DIRECTORY + 'IMG/' + df.iloc[0]['file_image_center']
    image_center = Image.open(file_image_center)
    image_center = np.asarray(image_center)
    image_shape = image_center.shape
    np_image = np.zeros((dataset_length, image_shape[0], image_shape[1], image_shape[2]))
    np_steer = np.zeros((dataset_length))
#    np_image = np.zeros((2*dataset_length, image_shape[0], image_shape[1], image_shape[2]))
#    np_steer = np.zeros((2*dataset_length))
    idx = list(range(dataset_length))
    random.shuffle(idx)
    for i in range(dataset_length):
        file_image_center = DATA_DIRECTORY + 'IMG/' + df.iloc[idx[i]]['file_image_center']
        steer = df.iloc[idx[i]]['steer']
        image_center         = np.array(Image.open(file_image_center).convert("RGB"))
#        image_center_flipped = np.array(Image.open(file_image_center).convert("RGB").transpose(Image.FLIP_LEFT_RIGHT))
        np_image[i,:,:,:] = image_center
        np_steer[i] =  float(steer)
#        np_image[i+1,:,:,:] = image_center_flipped
#        np_steer[i+1] = -float(steer)

    return np_image, np_steer

def get_data_generador(df, chunk_size = MINIBATCH_SIZE, shuffle = True):
    dataset_length = df.shape[0]
#    if dataset_length > 256:
#        dataset_length = 256
    print ('dataset_length', dataset_length)
    file_image_center = DATA_DIRECTORY + 'IMG/' + df.iloc[0]['file_image_center']
    image_center = Image.open(file_image_center)
    image_center = np.asarray(image_center)
    image_shape = image_center.shape
    idx_base_chunk = 0
    while True:
        np_image = np.zeros((chunk_size, image_shape[0], image_shape[1], image_shape[2]))
        np_steer = np.zeros((chunk_size))
        for i in range(chunk_size):
            file_image_center = DATA_DIRECTORY + 'IMG/' + df.iloc[i+idx_base_chunk]['file_image_center']
            steer = df.iloc[i+idx_base_chunk]['steer']
            image_center = np.array(Image.open(file_image_center).convert("RGB"))
            np_image[i+0,:,:,:] = image_center
            np_steer[i+0] =  float(steer)

        yield np_image, np_steer


def buildModel(image_shape):
    input_image = Input(shape = image_shape, name = 'image')
    x = input_image
    print ('x1', x)

    x = Lambda(lambda x: (x/255.0)-0.5)(x)

    print ('x2', x)

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
#    fc3 = Activation('relu')(fc3)
    predict_steer = Dense(1, name='predict_steer', kernel_initializer = keras.initializers.glorot_uniform())(fc3)
    model = Model(inputs = input_image, outputs = predict_steer, name='CarDriver')

    return model



x_train, y_train = get_data(df)
print ('x_train.shape', x_train.shape)
print ('y_train.shape', y_train.shape)

model = None

if os.path.isfile('CarDriver.h5'):
    model = load_model('CarDriver.h5')
else:
    model = buildModel((x_train.shape[1],x_train.shape[2],x_train.shape[3]))

print ('model', model)
model.summary()

sgd = optimizers.SGD(lr=0.0075, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='mean_squared_error')

callbacks_list = [keras.callbacks.ModelCheckpoint(filepath='CarDriver.h5', monitor='val_loss', save_best_only=False), Avisala_Callback()]


model.fit(x_train, y_train, epochs=40, batch_size=MINIBATCH_SIZE, validation_split=0.2, callbacks=callbacks_list)

#model.save('CarDriver.h5')
