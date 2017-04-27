import numpy as np
from load_data import load_data

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils

np.random.seed(1337)

# TRAIN A CONVNET

batch_size = 128
epochs = 16
classes = 7

# Input image dimensions
img_rows, img_col = 48, 48
# Number of convolutional filters
filters = 32
# Size of pooling area
pool = 2
# convolutional kernel size
conv = 3
