import keras
from PIL import Image
import skyrogue_loader
import numpy as np

RESOLUTION = (320, 240)

x = skyrogue_loader.load_images(RESOLUTION, skyrogue_loader.pixel_format.gray_minus_1_to_1)

you_died_original_size = Image.open("you_died.png")
you_died = Image.new('RGBA', RESOLUTION, (0, 0, 0, 0))
you_died.paste(you_died_original_size, box = (127, 96))

num_images = x.shape[0]
x = np.append(x, x)

for i in range(num_images, num_images*2):
    x[i].show(title = "original image")
    you_died_original_size.show(title = "augmenting bitmap")
    you_died.show(title = "augmented image")
    x[i] = Image.alpha_composite(you_died_bitmap, x[i])

