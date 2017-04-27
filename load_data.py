import numpy as np
import pickle
import os

DATA_PATH = './fer2013/fer2013.csv'
PICKLE_PATH = './pickleData'


def load_data():
    train_size = 28709
    test_size = 3589
    dim = 48
    X_train = np.empty([train_size, dim, dim])
    X_test = np.empty([test_size, dim, dim])
    y_train = np.empty(train_size)
    y_test = np.empty(test_size)

    file = open(DATA_PATH, 'r')

    train_index = test_index = 0
    for i, line in enumerate(file):
        if i >= 1:
            split_line = line.split(",")
            usage = split_line[2].rstrip()
            if usage == 'Training':
                X_train[train_index, :, :] = np.fromstring(split_line[1], dtype='int', sep=' ').reshape(dim, dim)
                y_train[train_index] = int(split_line[0])
                train_index += 1
            elif usage == 'PublicTest':
                X_test[test_index, :, :] = np.fromstring(split_line[1], dtype='int', sep=' ').reshape(dim, dim)
                y_test[test_index] = int(split_line[0])
                test_index += 1

    if not os.path.exists(PICKLE_PATH):
        os.makedirs(PICKLE_PATH)
    # Train model
    pickle.dump(X_train, open(PICKLE_PATH + '/fer_x_train', 'wb'))
    pickle.dump(y_train, open(PICKLE_PATH + '/fer_y_train', 'wb'))
    # Test model
    pickle.dump(X_test, open(PICKLE_PATH + '/fer_x_test', 'wb'))
    pickle.dump(y_test, open(PICKLE_PATH + '/fer_y_test', 'wb'))