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
#K.set_epsilon(1e-4)

BATCH_SIZE = 128
#BATCH_SIZE = 8
#NUM_EPOCH = 50
NUM_EPOCH = 50000
LR = 0.0002  # initial learning rate
B1 = 0.5  # momentum term
GENERATED_IMAGE_PATH = 'images/'
GENERATED_MODEL_PATH = 'models/'

profile = False

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
#        X_train = X_train[100:5100, :, :]
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

    if profile:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata= tf.RunMetadata()
    else:
        run_options = None
        run_metadata = None 


    # build GAN
    g = generator()
    # g = keras.models.load_model(GENERATED_MODEL_PATH+'dcgan_generator.h5')
    plot_model(g, to_file='generator.png')

    d = discriminator()
    # d = keras.models.load_model(GENERATED_MODEL_PATH+'dcgan_discriminator.h5')
    plot_model(d, to_file='discriminator.png')

    opt = Adam(lr=LR,beta_1=B1)
    d.compile(loss='binary_crossentropy',
              metrics=['accuracy'],
              optimizer=opt,
              options=run_options,
              run_metadata=run_metadata)

    d.trainable = False
    dcgan = Sequential([g, d])
    opt= Adam(lr=LR,beta_1=B1)
    dcgan.compile(loss='binary_crossentropy',
                  metrics=['accuracy'],
                  optimizer=opt,
                  options=run_options,
                  run_metadata=run_metadata)

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

    d_epoch_losses = []
    g_epoch_losses = []
    for epoch in list(map(lambda x: x+1, range(NUM_EPOCH))):
        # Add noise to the images
        image_fuzz = .5 ** (epoch / 30)
        #fuzz = 0

        # Use noisey labels
        label_fuzz = image_fuzz
        
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
            X_d_fuzz = X_d_true * (1 - image_fuzz) + label_fuzz * np.random.uniform(-1, 1, [BATCH_SIZE, 240, 320, 1])

            X_g = np.array([np.random.normal(0, 0.5, 100) for _ in range(BATCH_SIZE)])
            X_d_gen = g.predict(X_g, verbose=0)

            d_real_loss, d_real_acc = d.train_on_batch(X_d_fuzz, y_d_true)
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

        if profile:
            tl = timeline.Timeline(run_metadata.step_stats)
            ctf = tl.generate_chrome_trace_format()
            with open('timeline.json', 'w') as f:
                f.write(ctf)

        # Checkpoint the model and write save an image of the model's generated output
        if epoch % 10 == 0:
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
