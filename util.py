import math
import numpy as np

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

def combine_images(images):
    num_images = images.shape[0]
    image_height = images.shape[1]
    image_width = images.shape[2]
    x_images = int(math.sqrt(num_images))
    y_images = math.ceil(num_images / x_images)
    combined = np.zeros((image_height * y_images, image_width * x_images))

    for i in range(num_images):
        x = i % x_images
        y = i // x_images
        combined[y*image_height : (y+1)*image_height, x*image_width : (x+1)*image_width] = images[i][:, :, 0]
    return combined

def fix_rtx_bug():
    config = tf.ConfigProto()
#    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    set_session(sess)

