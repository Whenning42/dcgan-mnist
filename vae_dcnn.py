from PIL import Image
import keras
import model
import util
import skyrogue_loader
import os

BATCH_SIZE = 32

LR = .0002
B = .5

IMAGES_TO_OUTPUT = 49
OUTPUT_PATH = "autoencoder_results"

if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)

util.fix_rtx_bug()

# Loads data normalized from -1 to 1
x = skyrogue_loader.load_images()

LATENT_DIM = 128
encoder = model.encoder(x.shape[1:], LATENT_DIM)
decoder = model.decoder(x.shape[1:], LATENT_DIM)
autoencoder = keras.models.Sequential([encoder, decoder])
autoencoder.compile(loss = 'binary_crossentropy', optimizer = keras.optimizers.Adam(lr = LR, beta_1 = B))

def log_images(image_array, title):
    image = util.combine_images(image_array)
    image*127.5 + 127.5
    Image.fromarray(image.astype(np.uint8)).save(OUTPUT_PATH + title)

images_to_output = x[np.ranom.permutation(output_images)[0:output_images-1]]
log_images(images_to_output, "original_images.png")

for epoch in range(1, 1000):
    autoencoder.fit(x, x)
    log_images(autoencoder.predict(image_to_output), "%03depoch.png" % epoch)
