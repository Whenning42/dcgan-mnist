from PIL import Image
import keras
import model
import util
import skyrogue_loader
import os
import numpy as np
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm

BATCH_SIZE = 32

LR = .0002
B = .5

IMAGES_TO_OUTPUT = 49
OUTPUT_PATH = "autoencoder_output/"

if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)

util.fix_rtx_bug()

# Loads data normalized from -1 to 1
x = skyrogue_loader.load_images()
# x = x[0:4000]

LATENT_DIM = 64
encoder = model.encoder(x.shape[1:], LATENT_DIM)
#print("ES")
#print(encoder.summary())
decoder = model.decoder(x.shape[1:], LATENT_DIM)
#print("DS")
#print(decoder.summary())
autoencoder = keras.models.Sequential([encoder, decoder])
autoencoder.compile(loss = 'mse', optimizer = keras.optimizers.Adam())
#autoencoder.compile(loss = 'mse', optimizer = keras.optimizers.Adam(lr = LR, beta_1 = B))

def log_images(image_array, title):
    image = util.combine_images(image_array)
    image = image*127.5 + 127.5
    Image.fromarray(image.astype(np.uint8)).save(OUTPUT_PATH + title)

def print_error_histogram(data, autoencoder, title):
    enc = autoencoder.predict(data)
    losses = np.mean((data - enc)**2, axis=(1, 2, 3))
    plt.hist(losses, normed = True, bins = 100)
    plt.savefig(title)
    plt.clf()

images_to_output = x[np.random.permutation(x.shape[0])[0:49]]
log_images(images_to_output, "original_images.png")

for epoch in range(1, 1000):
    #history = autoencoder.fit(x, x, validation_split=.1, callbacks = [keras.callbacks.TensorBoard(histogram_freq = 1, write_grads = True)])
    history = autoencoder.fit(x, x)
#    print_error_histogram(x, autoencoder, "%03d_loss_hist.png" % epoch)
    print(history.history['loss'])
    log_images(autoencoder.predict(images_to_output), "%03depoch.png" % epoch)
    autoencoder.save(OUTPUT_PATH + "%03dautoencoder.h5" % epoch)
