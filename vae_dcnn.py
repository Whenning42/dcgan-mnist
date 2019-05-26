from PIL import Image
import keras
import model
import util
import config
import skyrogue_loader
import os
import numpy as np
import sys
import matplotlib.pyplot as plt

util.fix_rtx_bug()

# Logging configuration
IMAGES_TO_OUTPUT = 49
OUTPUT_PATH = "experiments/"
PRINT_ERROR_HISTOGRAM = False

if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)

IMAGE_RESOLUTION = (320, 240)
IMAGE_FORMAT = skyrogue_loader.pixel_format.gray_minus_1_to_1

# Loads data normalized from -1 to 1
x = skyrogue_loader.load_images(IMAGE_RESOLUTION, IMAGE_FORMAT)

for parameters in config.best_known_config.all_runs():
    number_of_runs = 1
    if ("trials" in parameters.keys()):
        number_of_runs = parameters["trials"]

    # Model hyperparameters
    BATCH_SIZE = parameters["BatchSize"]
    LR = parameters["LR"]
    B1 = parameters["Beta1"]
    B2 = parameters["Beta2"]
    LATENT_DIM = parameters["LatentDim"]
    # We're still missing the model architecture here

    for run in range(number_of_runs):
        image_shape = x.shape[1:]
        encoder = model.encoder(image_shape, LATENT_DIM)
        decoder = model.decoder(image_shape, LATENT_DIM)
        autoencoder = keras.models.Sequential([encoder, decoder])
        autoencoder.compile(loss = 'mse', optimizer = keras.optimizers.Adam(lr = LR, beta_1 = B1, beta_2 = B2))

        images_to_output = x[np.random.permutation(x.shape[0])[0:49]]
        log_images(images_to_output, "original_images.png")

        for epoch in range(1, 1000):
            history = autoencoder.fit(x, x)

            if (PRINT_ERROR_HISTOGRAM):
                print_error_histogram(x, autoencoder, "%03d_loss_hist.png" % epoch)
            print(history.history['loss'])
            log_images(autoencoder.predict(images_to_output), "%03depoch.png" % epoch)
            autoencoder.save(OUTPUT_PATH + "%03dautoencoder.h5" % epoch)
