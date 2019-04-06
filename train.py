#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from keras.datasets import mnist
from PIL import Image
from model import discriminator, generator
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from keras.utils import plot_model
from visualizer import *
from keras import backend as K
import keras
import skyrogue_loader

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

# After we have a seemingly working impl we can play around with fp16
#K.set_floatx('float16')
#K.set_epsilon(1e-3)

BATCH_SIZE = 32
#BATCH_SIZE = 8
#NUM_EPOCH = 50
NUM_EPOCH = 50000
LR = 0.0002  # initial learning rate
B1 = 0.5  # momentum term
GENERATED_IMAGE_PATH = 'images/'
GENERATED_MODEL_PATH = 'models/'



from PIL import Image

def train():
    cache_path = ".full_preprocessed.npy"
    try:
        X_train = np.load(cache_path)
        print("Got cached preproccessed images")
    except IOError:
        #(X_train, y_train), (_, _) = mnist.load_data()
        #(X_train, _), (_, _) = skyrogue.load_data(160, 120)
        X_train = skyrogue_loader.load_images()
        X_train = X_train[100:5100, :, :]
        # normalize images
        print("Started normalizing images")
        X_train = (X_train.astype(np.float16) - 127.5)/127.5
        print("Finished normalizing images")
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
        print("Finished reshape")
        np.save(cache_path, X_train)

    a = .2
    for i in range(1, X_train.shape[0], X_train.shape[0] // 20):
        continue
        print("Target")
        X_train[i, :, :, 0] = X_train[i, :, :, 0] * (1-a) + a * np.random.uniform(-1, 1, [240, 320])
        Image.fromarray(((X_train[i, :, :, 0] + 1) * 127.5).astype('uint8')).show()
        _ = input()

    # build GAN
    g = generator()
    plot_model(g, to_file='generator.png')

    d = discriminator()
    plot_model(d, to_file='discriminator.png')

    opt = Adam(lr=LR,beta_1=B1)
    d.compile(loss='binary_crossentropy',
              metrics=['accuracy'],
              optimizer=opt)

    d.trainable = False
    dcgan = Sequential([g, d])
    opt= Adam(lr=LR,beta_1=B1)
    dcgan.compile(loss='binary_crossentropy',
                  metrics=['accuracy'],
                  optimizer=opt)

    num_batches = int(X_train.shape[0] / BATCH_SIZE)
    # create directory
    if not os.path.exists(GENERATED_IMAGE_PATH):
        os.mkdir(GENERATED_IMAGE_PATH)
    if not os.path.exists(GENERATED_MODEL_PATH):
        os.mkdir(GENERATED_MODEL_PATH)

    print("-------------------")
    print("Total epoch:", NUM_EPOCH, "Number of batches:", num_batches)
    print("-------------------")
    #z_pred = np.array([np.random.uniform(-1,1,100) for _ in range(49)])
    z_pred = np.array([np.random.normal(0, 0.5, 100) for _ in range(49)])
    y_g = [1]*BATCH_SIZE
    y_d_true = [1]*BATCH_SIZE
    y_d_gen = [0]*BATCH_SIZE
    for epoch in list(map(lambda x: x+1, range(NUM_EPOCH))):
        fuzz = max((200 - epoch) / 200, 0)
    #    fuzz = 0
        
        shuffled_indices = np.random.permutation(X_train.shape[0])
        for index in range(num_batches):
            X_d_true = X_train[shuffled_indices[index*BATCH_SIZE:(index+1)*BATCH_SIZE]]
            X_d_fuzz = X_d_true * (1-fuzz) + fuzz * np.random.uniform(-1, 1, [BATCH_SIZE, 240, 320, 1])

            X_g = np.array([np.random.normal(0, 0.5, 100) for _ in range(BATCH_SIZE)])
            X_d_gen = g.predict(X_g, verbose=0)

            #h = d.predict(X_d_fuzz)
            dr_loss = d.train_on_batch(X_d_fuzz, y_d_true)
            #print()
            #print(h)
            #print(d.metrics_names)
            #print(K.eval(keras.losses.binary_crossentropy(K.variable(np.array(y_d_true)), K.variable(np.array(h[:, 0])))))

            df_loss = d.train_on_batch(X_d_gen, y_d_gen)
            g_loss = dcgan.train_on_batch(X_g, y_g)

            show_progress(epoch, index, g_loss[0], dr_loss[0], df_loss[0], g_loss[1], dr_loss[1], df_loss[1])

        # save generated images
        if epoch % 10 == 0:
            print("Fuzz factor:", fuzz)
            image = combine_images(g.predict(z_pred))
            image = image*127.5 + 127.5
            Image.fromarray(image.astype(np.uint8))\
                .save(GENERATED_IMAGE_PATH+"%03depoch.png" % (epoch))
            print()
            # save models
            g.save(GENERATED_MODEL_PATH+'dcgan_generator.h5')
            d.save(GENERATED_MODEL_PATH+'dcgan_discriminator.h5')

if __name__ == '__main__':
    train()
