# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense, Activation, Reshape, Cropping2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
#from BN16 import BatchNormalizationF16
from keras.layers.convolutional import UpSampling2D, Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU, ELU
from keras.optimizers import Adam
from keras.layers import Flatten, Dropout
import matplotlib.pyplot as plt


x_res = 320
y_res = 240

def generator(input_dim=128):
    model = Sequential()
    assert(x_res % 32 == 0)
    assert(y_res % 24 == 0)
    dn3 = (256, y_res//24, x_res//32)
    dn2 = (128, y_res//16, x_res//16)
    dn1 = (64, y_res//8, x_res//8)
    d0 = (32, y_res//4, x_res//4)
    d1 = (16, y_res//2, x_res//2)

    model.add(Dense(dn3[0] * dn3[1] * dn3[2], input_dim = input_dim))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Reshape((dn3[1], dn3[2], dn3[0]), input_shape = (dn3[0] * dn3[1] * dn3[2],)))
    model.add(UpSampling2D((2, 2)))

    for dims in [dn2, dn1, d0, d1]:
        model.add(Conv2D(dims[0], (5, 5), padding='same'))
        model.add(BatchNormalization())
        model.add(ELU())
        model.add(UpSampling2D((2, 2)))

    model.add(Conv2D(1, (5, 5), padding='same'))

    model.add(Cropping2D(((320-240) // 2, 0)))

    model.add(Activation('tanh'))
#    print(model.summary())
    return model

def decoder(input_shape, latent_dims):
    assert(input_shape == (240, 320, 1))
    return generator(latent_dims)

# Repeats a lot of code with discriminator
def encoder(input_shape, latent_dims):
    nb_filter = 8
    assert(input_shape == (240, 320, 1))

    model = Sequential()
    model.add(Conv2D(nb_filter, (5, 5), strides=(2, 2), padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(ELU())

    for i in range(len([32, 64, 128, 256])):
        model.add(Conv2D(min(2**(i+1) * nb_filter, 256), (5, 5), strides=(2, 2)))
        model.add(BatchNormalization())
        model.add(ELU())

    model.add(Flatten())
    model.add(Dense(512))
    model.add(ELU())
    model.add(Dense(latent_dims))
    model.add(Activation('tanh'))
    return model

def discriminator(input_shape=(y_res, x_res, 1), nb_filter = 8):
    model = Sequential()

    model.add(Conv2D(nb_filter, (5, 5), strides=(2, 2), padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(ELU())

    for i in range(len([32, 64, 128, 256])):
        model.add(Conv2D(min(2**(i+1) * nb_filter, 256), (5, 5), strides=(2, 2)))
        model.add(BatchNormalization())
        model.add(ELU())

    model.add(Flatten())
    model.add(Dense(512))
    model.add(ELU())
    model.add(Dense(128))
    model.add(ELU())
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
 #   print(model.summary())
    return model

