import os
import random
from PIL import Image
from tqdm import tqdm
import numpy as np
import numexpr as ne
from enum import Enum

cache_path = ".full_preprocessed.npy"

class pixel_format(Enum):
    rgb_256 = 1
    gray_minus_1_to_1 = 2

def load_images(resolution, format):
    assert(resolution == (320, 240))
    assert(format == pixel_format.gray_minus_1_to_1)

    try:
        images = np.load(cache_path)
    except IOError:
        # Load the images
        print("Didn't find cached images")
        folder = "2019-04-01 20:22:11:109813"

        num_images = len(os.listdir(folder)) // 2
        images = np.zeros([num_images, 240, 320], dtype='uint8')

        i = 0
        for file in tqdm(sorted(os.listdir(folder))):
            if file[-3:] != "png":
                continue

            image_path = folder + "/" + file
            image = Image.open(image_path).convert('L').resize([320, 240])
            images[i] = np.array(image)
            i += 1

        assert(i == num_images)

        # Normalize the images
        images = images.astype('float32')
        images = ne.evaluate('(images - 127.5)/127.5')
        print("Finished normalizing images")
        images = images.reshape(images.shape[0], images.shape[1], images.shape[2], 1)
        print("Finished reshape")
        np.save(cache_path, images)

    return images
