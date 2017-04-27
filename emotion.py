import numpy as np
from load_data import load_data

np.random.seed(1337)

# TRAIN A CONVNET

nb_batch_size = 128
nb_epochs = 16
nb_classes = 7

# Input image dimensions
img_rows, img_col = 48, 48
# Number of convolutional filters
nb_filters = 32
# Size of pooling area
nb_pool = 2
# convolutional kernel size
nb_conv = 3
