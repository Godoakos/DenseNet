# -*- encoding: utf-8 -*-

import numpy as np
import PIL
import matplotlib.pyplot as plt

def normalize_img(img):
    mean = np.mean(img)
    img = np.subtract(img, mean)
    var = np.var(img)
    img /= var
    return img