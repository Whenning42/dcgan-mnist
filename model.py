# -*- coding: utf-8 -*-
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Reshape, Cropping2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
#from BN16 import BatchNormalizationF16
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU, ELU
from keras.optimizers import Adam
from keras.layers import Flatten, Dropout
import matplotlib.pyplot as plt

import numpy as np
import keras.backend as K

x_res = 320
y_res = 240

NEW_CONV_END = False
NEW_CONV_START = False

RELU_SLOPE = .2
USE_ATTENTION = False

# Concatenates the input to the current activations, forming a skip connection
def skip_concat(current, input, shape):
    out_dims = shape[0] * shape[1] * shape[2]
    x = Dense(out_dims)(input)
    x = LeakyReLU(RELU_SLOPE)(x)
    x = Reshape((shape[1], shape[2], shape[0]), input_shape = (out_dims,))(x)
    return keras.layers.Concatenate()([current, x])

# A function to be used as a lambda below in the coords_concat function.
def concat(x):
    current = x
    shape = (x.shape[1], x.shape[2])
    coords = np.array(np.meshgrid(np.linspace(-1, 1, shape[1]), np.linspace(-1, 1, shape[0])))
    coords = np.moveaxis(coords, 0, -1)
    coords = np.expand_dims(coords, 0) # Add a batch dimension
    coords = K.variable(coords)
    coords = K.tile(coords, (K.shape(current)[0], 1, 1, 1)) # Tile along the batch dimension
    return keras.layers.Lambda(lambda x: K.concatenate(x))([current, coords])


# Concatenates the xy coordinates of pixels to the currenct activations.
# This *might* help with the accuracy of UI elements in images, but I haven't conclusively tested this.
def coords_concat(current, shape):
    return keras.layers.Lambda(concat)(current)

def generator(input_dim = 128):
    assert(x_res == 320)
    assert(y_res == 240)

    slices = [(10, 10, 256),
              (20, 20, 128),
              (40, 40, 64),
              (80, 80, 32),
              (160, 160, 16),
              (320, 320, 1)]

    input = keras.layers.Input(shape = (input_dim,))

    res = Dense(input_dim)(input)
    res = LeakyReLU(RELU_SLOPE)(res)
    x = keras.layers.Add()([x, res])

    slice_zero_total_dim = np.prod(slices[0])
    x = Dense(slize_zero_total_dim, input_dim = input_dim)(x)
    x = LeakyReLU(RELU_SLOPE)(x)

    x = Reshape(silces[0], input_shape = (slice_zero_total_dim,))(x)
    if (NEW_CONV_START):
        for i in Range(0, len(slices) - 1):
            x = Conv2D(slices[i][2], (5, 5), padding='same')(x)
            x = LeakyReLU(RELU_SLOPE)(x)
            x = UpSampling2D((2, 2))(x)
    else:
        x = UpSampling2D((2, 2))(x)

        for i in Range(1, len(slices) - 1):
            x = Conv2D(slices[i][2], (5, 5), padding='same')(x)
            x = LeakyReLU(RELU_SLOPE)(x)
            x = UpSampling2D((2, 2))(x)

    if (NEW_CONV_END):
        x = Conv2D(4, (5, 5), padding='same')(x)
        x = LeakyReLU(RELU_SLOPE)(x)
        x = Conv2D(1, (5, 5), padding='same')(x)
    else:
        x = Conv2D(1, (5, 5), padding='same')(x)

    x = Cropping2D(((x_res - y_res) // 2, 0))(x)
    x = Activation('tanh')(x)

    return keras.models.Model(inputs = input, outputs = x)

def decoder(input_shape, latent_dims):
    if (USE_ATTENTION):
        assert(input_shape == (240, 320, 1))
        x_0 = generator(latent_dims)
        x_1 = generator(latent_dims)
        a_0 = generator(latent_dims)
        a_1 = generator(latent_dims)

        a = keras.layers.concatenate([a_0, a_1])
        a = keras.layers.Lambda(lambda x: K.softmax(x))([a])

        x = keras.layers.concatenate([x_0, x_1])
        x = keras.layers.multiply([a, x])
        x = keras.layers.Lambda(lambda x: K.sum(x, axis = -1, keepdims = True))([x])
        return x

    else:
        assert(input_shape == (240, 320, 1))
        return generator(latent_dims)

# Repeats a lot of code with discriminator
def encoder(input_shape, latent_dims):
    nb_filter = 6
    assert(input_shape == (240, 320, 1))

    input = keras.layers.Input(shape = input_shape)
    x = input

    x = Conv2D(nb_filter, (5, 5), strides=(2, 2), padding='same', input_shape=input_shape)(x)
    # x = BatchNormalization()(x)
    x = LeakyReLU(RELU_SLOPE)(x)

    for channels in [nb_filter * 2 ** (i+1) for i in range(4)]:
#        x = coords_concat(x, (-1, -1))
        x = Conv2D(channels, (5, 5), strides=(2, 2))(x)
        # x = BatchNormalization()(x)
        x = LeakyReLU(RELU_SLOPE)(x)

    x = Flatten()(x)
    x = Dense(512)(x)
    x = LeakyReLU(RELU_SLOPE)(x)

    # x = Dense(256)(x)  ## Should remove
    # x = LeakyReLU(.1)(x)

    ## Also change latent_dims in vae_dcgan.py

    x = Dense(latent_dims)(x)
    x = Activation('tanh')(x)

    return keras.models.Model(inputs = input, outputs = x)

def discriminator(input_shape=(y_res, x_res, 1), nb_filter = 32):
    model = Sequential()

    model.add(Conv2D(nb_filter, (5, 5), strides=(2, 2), padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(LeakyReLU(RELU_SLOPE))

    for i in range(len([32, 64, 128, 256])):
        model.add(Conv2D(2**(i+1) * nb_filter, (5, 5), strides=(2, 2)))
        model.add(BatchNormalization())
        model.add(LeakyReLU(RELU_SLOPE))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(LeakyReLU(RELU_SLOPE))
    model.add(Dense(128))
    model.add(LeakyReLU(RELU_SLOPE))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
 #   print(model.summary())
    return model

