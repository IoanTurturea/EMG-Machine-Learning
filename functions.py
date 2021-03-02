import numpy as np
import os
import csv
import tensorflow as tf
from tensorflow import keras

path = "/home/ioan/Desktop/Database/"



def get_accuracy_from_confusion_matrix(confusion_matrix):
    acc = 0.0
    for k in range(confusion_matrix.shape[0]):
        acc += confusion_matrix[k][k]

    return 100.0*acc/np.sum(confusion_matrix)


def create_confusion_matrix(label_test, label_predict):
    label_test = np.asanyarray(label_test, dtype=int)
    label_predict = np.asanyarray(label_predict, dtype=int)
    nr_1 = int(label_test.max() + 1)
    nr_t = label_test.shape[0]
    confusion_matrix = np.zeros((nr_1, nr_1), dtype=np.int32)
    for i in range(nr_t):
        confusion_matrix[label_test[i]][label_predict[i]] += 1

    return confusion_matrix


# extracts data from files of database
# located in directory: "/home/ioan/Desktop/Database/"
# makes a data augmentation by overlapping
# with a window of size 50%
# returns 6 tensorflow.data.Dataset objects:
# x_train, y_train, x_val, y_val, x_test, y_test
# Obs: val is the same with dev
def prepare_xy_train_val_test_asnumpyarrays():
    # data base read and format:
    x_train, y_train = [], []
    x_test, y_test = [], []
    x_val, y_val = [], []

    # window size, choose it to be 50 samples
    # because 50 & 200Hz(sampling rate) = 250ms
    # a sample = one line from a (*,8) matrix
    N = 50
    overlap = 40
    hamming = np.hamming(N*8) # did not improve accuracy :(
    # hamming = np.reshape(hamming, (N*8, )) # reshape both window if wanted, but does not make improvements

    folders = os.listdir(path) # there are 3 folders in Database: Train, Val, Test

    # iterate in folders
    for folder in folders:
        data = os.path.join(path, folder)
        files = os.listdir(data)

        # iterate files in a folder
        for file in files:
            filepath = data + "/" + str(file)
            signal = np.loadtxt(filepath)
            label = int((str(file))[5:7])

            step = N - overlap
            real_max_value = len(signal) - overlap
            max_value = int(real_max_value / step) * step # multiples of overlap must fit signal length.

            # Here was a minor bug: windows dimensions not equal to last window dimension for overlap != 25
            for i in range(0, max_value, step):
                window = signal[i: (i + N)]
                window = np.reshape(window, (N * 8))
                # window = np.reshape(window, (N*8, )) # reshape both hamming if wanted, but does not make improvements
                # window *= hamming

                if folder == 'Train':
                    x_train.append(window)
                    y_train.append(label)
                elif folder == 'Test':
                    x_test.append(window)
                    y_test.append(label)
                elif folder == 'Val':
                    x_val.append(window)
                    y_val.append(label)

    x_train = np.asanyarray(x_train)
    y_train = np.asanyarray(y_train)
    x_val = np.asanyarray(x_val)
    y_val = np.asanyarray(y_val)
    x_test = np.asanyarray(x_test)
    y_test = np.asanyarray(y_test)

    # make one hot encodings
    y_train = keras.utils.to_categorical(y_train, 13)
    y_val = keras.utils.to_categorical(y_val, 13)
    y_test = keras.utils.to_categorical(y_test, 13)

    return x_train, y_train, x_val, y_val, x_test, y_test
