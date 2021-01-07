import numpy as np
import os
import csv
import tensorflow as tf
from tensorflow import keras

# extracts data from files of database
# located in directory: "/home/ioan/Desktop/Database/"
# makes a data augmentation by overlapping
# with a window of size 50%


path = "/home/ioan/Desktop/Database/"

def prepare_xy_train_val_test_asnumpyarrays():
    # data base read and format:
    x_train, y_train = [], []
    x_test, y_test = [], []
    x_val, y_val = [], []

    # window size, choose it to be 50 samples
    # because 50 & 200Hz(sampling rate) = 250ms
    # a sample = one line from a (*,8) matrix
    N = 50
    overlap = 25

    # there are 3 folders in Database: Train, Val, Test
    folders = os.listdir(path)

    # iterate in folders
    for folder in folders:
        data = os.path.join(path, folder)
        files = os.listdir(data)

        # iterate files in a folder
        for file in files:
            filepath = data + "/" + str(file)
            signal = np.loadtxt(filepath)
            label = int((str(file))[5:7])
            # here was the problem:
            # returns size = rows*columns. Is this desired?
            # length = signal.size
            # instead use "len" attribute
            # N - overlap = step
            for i in range(0, len(signal) - N + 1, N - overlap):
                window = signal[i: (i + N)]

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

    # reshape x_train, x_val, x_test
    #x_train = np.reshape(x_train, (int(np.size(x_train) / 400), 400))
    x_train = np.reshape(x_train, (len(x_train), 400))
    print(x_train.shape)
    x_val = np.reshape(x_val, (len(x_val), 400))
    print(x_val.shape)
    x_test = np.reshape(x_test, (len(x_test), 400))
    print(x_test.shape)

    # reshape y_train, y_val, y_test
    y_train = np.reshape(y_train, (len(y_train), 1))
    print(y_train.shape)
    y_val = np.reshape(y_val, (len(y_val), 1))
    print(y_val.shape)
    y_test = np.reshape(y_test, (len(y_test), 1))
    print(y_test.shape)

    # make one hot encodings
    y_train = keras.utils.to_categorical(y_train, 13)
    y_val = keras.utils.to_categorical(y_val, 13)
    y_test = keras.utils.to_categorical(y_test, 13)

    return x_train, y_train, x_val, y_val, x_test, y_test

