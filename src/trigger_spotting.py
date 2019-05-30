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
    x_rgba = np.concatenate((loaded, loaded, loaded, loaded*0 + 1), axis = 2)
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
#    x[i, :, :, 0] = (np.array(x_im.convert('L')) - 127.7) / 127.5
#
#net = model.simple_conv(x[0].shape)
#net.compile(loss = 'binary_crossentropy', optimizer = keras.optimizers.Adam(lr = .0005), metrics = [keras.metrics.binary_accuracy])
#
#net.fit(x, y, epochs = 2)

# Training a model to spot fixed length text in a frame
# We should maybe check that the dollar sign is in the right spot before augmenting images
MAX_DIGITS = 4
num_digits = np.random.randint(0, MAX_DIGITS + 1, size = num_images)
# num_digits = np.random.randint(1, 2, size = num_images)

DIGIT_TYPES = 11
y = np.zeros((num_images, MAX_DIGITS, DIGIT_TYPES))
FONT_PATH = "fonts/clean_bold.pil"
COPY_SLICE = (15, 189, 48, 198)
TEXT_UPPER_LEFT = (15, 199)
PASTE_SLICE = (15, 199, 48, 208)
SLICE_WIDTH = COPY_SLICE[2] - COPY_SLICE[0]
SLICE_HEIGHT = COPY_SLICE[3] - COPY_SLICE[1]
for i in tqdm(range(0, num_images)):
    if np.mean(x[i, 80:240, 60:180, 0]) > .99:
        # This is a loading which has no contrast with white digits
        # so don't add digits and label it as having no digits
        y[i, :, 10] = 1
        continue

    # Pick string
    digits = np.full((MAX_DIGITS,), 10)
    digits[:num_digits[i]] = np.random.randint(0, DIGIT_TYPES - 1, size = num_digits[i])
    y[i, np.arange(MAX_DIGITS), digits] = 1
    string = ""
    for d in digits:
        if d != 10:
            string += str(d)

    # Copy background scene over possibly present player money UI element
    x_im = rgba_image_from_loaded(x[i])
    x_im_background_grab = x_im.crop(COPY_SLICE)
    x_im_background_grab.load()
    x_im.paste(x_im_background_grab, box = TEXT_UPPER_LEFT)

    # Render generated player money
    text_overlay = Image.new("RGBA", RESOLUTION, (0, 0, 0, 0))
    text_draw = ImageDraw.Draw(text_overlay)
    font = ImageFont.load(FONT_PATH)
    text_draw.text(TEXT_UPPER_LEFT, string, (255, 255, 255, 255), font = font)
    x_im = Image.alpha_composite(x_im, text_overlay)
    x_im = x_im.crop(PASTE_SLICE).resize(RESOLUTION)

    x[i, :, :, 0] = (np.array(x_im.convert('L')) - 127.7) / 127.5

# OUT = 1997
# print(x.shape)
# x[:, :, :, :] = x[OUT, :, :, :]
# print(x.shape)
# print(y.shape)
# y[:, :, :] = y[OUT, :, :]
# print(y.shape)

net = model.simple_one_hot_conv(x[0].shape, (MAX_DIGITS, DIGIT_TYPES))
net.compile(loss = 'categorical_crossentropy', optimizer = keras.optimizers.Adam(lr = .001), metrics = [keras.metrics.categorical_accuracy])

net.fit(x, y, epochs = 2)

### Loss test
# from keras import backend as K
# label = [7, 11, 11, 11]
# label = np.array([keras.utils.to_categorical(label)])
# w_on = np.array([.09, .5, .8, .9])
# w_off = (1 - w_on) / 10
# predicted = np.array([[[w_on[i] * label[0][i][j] + w_off[i] * (1 - label[0][i][j]) for j in range(label.shape[2])] for i in range(label.shape[1])]])
# print(label)
# print(predicted)
# y_true = K.variable(label)
# y_pred = K.variable(predicted)
# error = K.eval(keras.losses.categorical_crossentropy(y_true, y_pred))
# print(error)
# exit()
###

# np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
# for i in range(20):
#     out = random.randint(0, x.shape[0])
#     x_im = rgba_image_from_loaded(x[out])
#     y_pred = net.predict(x[out : out + 1])[0]
#     y_true = y[out]
#     print(net.evaluate(x[out : out + 1], y[out : out + 1]))
#     print(y_pred)
#     print(y_true)
#     x_im.show()
#     a = input()
