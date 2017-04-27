from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adadelta

# Input image dimensions
img_rows, img_cols = 48, 48
# Number of convolutional filters
nb_filters = 32
# Size of pooling area
nb_pool = 2
# convolutional kernel size
nb_conv = 3
nb_classes = 7

model = Sequential()

model.add(Convolution2D(nb_filters, nb_conv, nb_conv, input_shape=(img_rows, img_cols, 1)))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adadelta(), metrics=['accuracy'])
model.save('FerModel.h5')