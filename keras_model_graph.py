
import keras
from keras.layers import Activation, Input, Dense, Embedding, LSTM, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Concatenate, Dropout
from keras.models import Model
from keras.models import load_model
from keras.layers.convolutional import Cropping2D
from keras.layers.core import Lambda

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model


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

image_shape = (160, 320, 3)
model = buildModel(image_shape)
model.summary()

SVG(model_to_dot(model).create(prog='dot', format='svg'))
plot_model(model, to_file = 'model.png')
