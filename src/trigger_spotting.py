import keras
from PIL import Image, ImageFont, ImageDraw
import skyrogue_loader
import numpy as np
import model
import util
import random
from tqdm import tqdm

util.fix_rtx_bug()

RESOLUTION = (320, 240)
FORMAT = skyrogue_loader.pixel_format.gray_minus_1_to_1
x = skyrogue_loader.load_images(RESOLUTION, FORMAT)
x = x[:10000]
num_images = x.shape[0]

# pngs and bounding boxes come from photoshop
TEMPLATE_PATH = "templates/"
you_died_spec = {"path": "you_died.png", "upper left": (127, 96)}
mission_complete_spec = {"path": "mission_complete.png", "upper left": (65, 112)}
dollar_sign_spec = {"path": "dollar_sign.png", "upper left": (5, 199)}

# Converts from the format skyrogue loader loads into a PIL format
# This could get rolled into the skyrogue loader
def rgba_image_from_loaded(loaded):
    assert(FORMAT == skyrogue_loader.pixel_format.gray_minus_1_to_1)
    x_rgba = np.concatenate((x[i], x[i], x[i], x[i]*0 + 1), axis = 2)
    x_rgba = (x_rgba * 127.5 + 127.5).astype('uint8')
    return Image.fromarray(x_rgba).convert('RGBA')

# Train a model to spot a specific image in a frame
#spec = dollar_sign_spec
#original_size = Image.open(TEMPLATE_PATH + spec["path"])
#y = np.random.choice([1, 0], size = num_images, p = [.5, .5])
#
#for i in tqdm(range(0, num_images)):
#    if y[i] == 0:
#        continue
#    if np.mean(x[i, 80:240, 60:180, 0]) > .99:
#        # This is a loading screen so don't augment it
#        y[i] = 0
#        continue
#    x_im = rgba_image_from_loaded(x[i])
#
#    dx = random.randint(-10, 10)
#    dy = random.randint(-10, 10)
#
#    overlay = Image.new('RGBA', RESOLUTION, (0, 0, 0, 0))
#    overlay.paste(original_size, box = (spec["upper left"][0] + dx, spec["upper left"][1] + dy))
#
#    x_im = Image.alpha_composite(x_im, overlay)
#    x[i, :, :, 0] = np.array(x_im.convert('L'))
#
#net = model.simple_conv(x[0].shape)
#net.compile(loss = 'binary_crossentropy', optimizer = keras.optimizers.Adam(lr = .0005), metrics = [keras.metrics.binary_accuracy])
#
#net.fit(x, y, epochs = 2)

# Training a model to spot fixed length text in a frame
# We should maybe check that the dollar sign is in the right spot before augmenting images
MAX_DIGITS = 4
num_digits = np.random.randint(0, MAX_DIGITS + 1, size = num_images)


DIGIT_TYPES = 11
y = np.zeros((num_images, MAX_DIGITS, DIGIT_TYPES))
FONT_PATH = "fonts/"
fonts = ["clean.pil"]
COPY_SLICE = (15, 189, 48, 198)
TEXT_UPPER_LEFT = (15, 199)
for i in tqdm(range(0, num_images)):
    # Pick string
    digits = np.full((MAX_DIGITS,), 10)
    digits[:num_digits[i]] = np.random.randint(0, DIGIT_TYPES - 1, size = num_digits[i])
    y[i, np.arange(MAX_DIGITS), digits] = 1
    string = ""
    for d in digits:
        if d != 10:
            string += str(d)
    print(string)

    # Copy background scene over possibly present player money UI element
    x_im = rgba_image_from_loaded(x[i])
    x_im_background_grab = x_im.crop(COPY_SLICE)
    x_im_background_grab.load()
    x_im.paste(x_im_background_grab, box = TEXT_UPPER_LEFT)
    x[i, :, :, 0] = np.array(x_im.convert('L'))

    # Render generated player money
