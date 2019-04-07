import os
import random
from PIL import Image
from tqdm import tqdm
import numpy as np

def load_images():
    cached_np_path = ".cached_images.npy"
    try:
        images = np.load(cached_np_path)
        print("Finished loading images")
        return images
    except IOError:
        print("Didn't find cached images")
        folder = "2019-04-01 20:22:11:109813"
        num_images = len(os.listdir(folder)) // 2
        images = np.zeros([num_images, 240, 320], dtype='uint8')
        i=0
        for file in tqdm(sorted(os.listdir(folder))):
            if file[-3:] != "png":
                continue
              
            image_path = folder + "/" + file
            image = Image.open(image_path).convert('L').resize([320, 240])
            if i == 500:
                image.show()
            images[i] = np.array(image)
            i += 1
        np.save(cached_np_path, images)

        return images
