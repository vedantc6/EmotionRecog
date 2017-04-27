import numpy as np
from keras.models import load_model
import pickle
from keras.utils import np_utils

np.random.seed(1337)

# TRAIN A CONVNET
nb_batch_size = 128
nb_epochs = 16
nb_classes = 7

# Input image dimensions
img_rows, img_cols = 48, 48
# Number of convolutional filters
nb_filters = 32
# Size of pooling area
nb_pool = 2
# convolutional kernel size
nb_conv = 3

def c_neural_network():
    FerModel = load_model('FerModel.h5')
    return FerModel


def train_model(model):
    X_train = pickle.load(open('./pickleData/fer_x_train', 'rb'))
    y_train = pickle.load(open('./pickleData/fer_y_train', 'rb'))
    X_test = pickle.load(open('./pickleData/fer_x_test', 'rb'))
    y_test = pickle.load(open('./pickleData/fer_y_test', 'rb'))

    # print(X_train.shape)

    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")
    X_train /= 255
    X_test /= 255

    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # Converting class vectors to binary class matrices
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)

    print('Y_train shape:', y_train.shape)

    model.fit(X_train, y_train,
              batch_size=nb_batch_size,
              nb_epoch=nb_epochs,
              shuffle=True,
              verbose=1,
              validation_data=(X_test, y_test))
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print(('Test accuracy:', score[1]))

train_model(c_neural_network())