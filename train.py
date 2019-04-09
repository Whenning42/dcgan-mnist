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
from tensorflow.python.client import timeline
import matplotlib.pyplot as plt
import numexpr as ne

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
#K.set_epsilon(5e-4)

BATCH_SIZE = 128
#BATCH_SIZE = 8
#NUM_EPOCH = 50
NUM_EPOCH = 50000
LR_G = 0.0002  # initial learning rate
LR_D = 0.0002  # initial learning rate
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
        X_train = skyrogue_loader.load_images().astype(np.float32)
        #X_train = X_train[100:5100, :, :]
        # normalize images
        print("Started normalizing images")
        X_train = ne.evaluate('(X_train - 127.5)/127.5')
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
    # g = keras.models.load_model(GENERATED_MODEL_PATH+'dcgan_generator.h5')
    plot_model(g, to_file='generator.png')

    d = discriminator()
    # d = keras.models.load_model(GENERATED_MODEL_PATH+'dcgan_discriminator.h5')
    plot_model(d, to_file='discriminator.png')

    opt = Adam(lr=LR_D, beta_1=B1)
    d.compile(loss='binary_crossentropy',
              metrics=['accuracy'],
              optimizer=opt)

    d.trainable = False
    dcgan = Sequential([g, d])
    opt= Adam(lr=LR_G, beta_1=B1)
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
    z_pred = np.array([np.random.normal(0, 0.5, 128) for _ in range(49)])

    # Pre-generate noise to speed up the train loop
    NOISE_IMAGES = 2000
    noise_images = np.random.uniform(-1, 1, [NOISE_IMAGES, 240, 320, 1])

    d_epoch_losses = []
    g_epoch_losses = []
    for epoch in list(map(lambda x: x+1, range(NUM_EPOCH))):
        # Add noise to the images
        image_fuzz = .5 ** (epoch / 10) * .5
        #fuzz = 0

        # Use noisey labels
        label_fuzz = .05
        
        d_batch_losses = []
        g_batch_losses = []

        shuffled_indices = np.random.permutation(X_train.shape[0])

        for index in range(num_batches):
            y_g = np.ones(BATCH_SIZE)
            y_d_true = np.ones(BATCH_SIZE)
            y_d_gen = np.zeros(BATCH_SIZE)
            for arr in [y_g, y_d_true, y_d_gen]:
                random_label = np.random.choice([0, 1], BATCH_SIZE, p=[.5, .5])
                label_mask = np.random.choice([0, 1], BATCH_SIZE, p=[1 - label_fuzz, label_fuzz]) == 1
                arr[label_mask] = random_label[label_mask]

            X_d_true = X_train[shuffled_indices[index*BATCH_SIZE:(index+1)*BATCH_SIZE]]
            noise_index = np.random.randint(0, NOISE_IMAGES - BATCH_SIZE)
            noise_to_use = noise_images[noise_index : noise_index + BATCH_SIZE]
            X_fuzz = ne.evaluate('X_d_true * (1-image_fuzz) + image_fuzz * noise_to_use')

            X_g = np.array([np.random.normal(0, 0.5, 128) for _ in range(BATCH_SIZE)])
            X_d_gen = g.predict(X_g, verbose=0)

            d_real_loss, d_real_acc = d.train_on_batch(X_fuzz, y_d_true)
            d_generated_loss, d_generated_acc = d.train_on_batch(X_d_gen, y_d_gen)
            d_batch_losses.append((d_generated_loss + d_real_loss) / 2)

            g_loss, g_acc = dcgan.train_on_batch(X_g, y_g)
            g_batch_losses.append(g_loss)

            show_progress(epoch, index, g_loss, d_real_loss, d_generated_loss,
                          g_acc, d_real_acc, d_generated_acc)

        plt.clf()
        d_epoch_losses.append(np.mean(d_batch_losses))
        g_epoch_losses.append(np.mean(g_batch_losses))
        plt.plot(d_epoch_losses, label="Discriminator loss")
        plt.plot(g_epoch_losses, label="Generator loss")
        plt.legend()
        
        plt.yscale("linear")
        plt.savefig('loss_chart.png')

        plt.yscale("log")
        plt.savefig('log_loss_chart.png')

        # Checkpoint the model and write save an image of the model's generated output
        if epoch % 5 == 0:
            print()
            print("Fuzz factor:", image_fuzz)
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
