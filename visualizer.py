# -*- coding: utf-8 -*-
import math
import numpy as np
import sys

def combine_images(generated_images):
    total,height,width = generated_images.shape[:-1]
    cols = int(math.sqrt(total))
    rows = math.ceil(float(total)/cols)
    combined_image = np.zeros((height*rows, width*cols),
                              dtype=generated_images.dtype)

    for index, image in enumerate(generated_images):
        i = int(index/cols)
        j = index % cols
        combined_image[height*j:height*(j+1), width*i:width*(i+1)] = image[:, :, 0]
    return combined_image

def show_progress(e, i, g0, dr0, df0, g1, dr1, df1):
    sys.stdout.write("\repoch: %d, batch: %d, g_loss: %f, dr_loss: %f, df_loss: %f, g_accuracy: %f, dr_accuracy: %f, df_accuracy: %f" % (e, i, g0, dr0, df0, g1, dr1, df1))
    sys.stdout.flush()

