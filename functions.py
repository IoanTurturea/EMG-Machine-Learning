import numpy as np
import os
import csv
import tensorflow as tf


# extracts data from files of database
# located in directory: "/home/ioan/Desktop/Database/"
# makes a data augmentation by overlapping
# with a window of size 50%
# returns 3 tensorflow.data.Dataset pair objects:
# train_dataset, val_dataset, test_dataset
from tensorflow import keras


def prepare_train_val_test():
    # data base read and format:
    x_train, y_train = [], []
    x_test, y_test = [], []
    x_val, y_val = [], []

    path = "/home/ioan/Desktop/Database/"

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
            label = (str(file))[5:7]
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

    # old implementation (before dec. 4)
    # train_dataset = tf.data.Dataset.from_tensor_slices((np.asanyarray(x_train), np.asanyarray(y_train)))
    # val_dataset = tf.data.Dataset.from_tensor_slices((np.asanyarray(x_val), np.asanyarray(y_val)))
    # test_dataset = tf.data.Dataset.from_tensor_slices((np.asanyarray(x_test), np.asanyarray(y_test)))

    # 4 dec. update:
    # left away the old return implementation
    # because it looks easier to make
    # one_hot encoding and batch from this function
    # rather than breaking the already zipped dataset object in
    # data and label at the function call
    # Be aware of the one_hot and batch size!
    # The lambda expression can not be compiled!

    xtrain = tf.data.Dataset.from_tensor_slices(np.asanyarray(x_train))
    ytrain = tf.data.Dataset.from_tensor_slices(np.asanyarray(y_train))  # .map(lambda z: tf.one_hot(z, 13))
    train_dataset = tf.data.Dataset.zip((xtrain, ytrain)).repeat().batch(16)

    xval = tf.data.Dataset.from_tensor_slices(np.asanyarray(x_val))
    yval = tf.data.Dataset.from_tensor_slices(np.asanyarray(y_val))  # .map(lambda z: tf.one_hot(z, 13))
    val_dataset = tf.data.Dataset.zip((xval, yval)).repeat().batch(16)

    xtest = tf.data.Dataset.from_tensor_slices(np.asanyarray(x_test))
    ytest = tf.data.Dataset.from_tensor_slices(np.asanyarray(y_test))  # .map(lambda z: tf.one_hot(z, 13))
    test_dataset = tf.data.Dataset.zip((xtest, ytest)).repeat().batch(16)

    return train_dataset, val_dataset, test_dataset


# extracts data from files of database
# located in directory: "/home/ioan/Desktop/Database/"
# makes a data augmentation by overlapping
# with a window of size 50%
# returns 6 tensorflow.data.Dataset objects:
# x_train, y_train, x_val, y_val, x_test, y_test
# Obs: val is the same with dev
def prepare_xy_train_val_test():
    # data base read and format:
    x_train, y_train = [], []
    x_test, y_test = [], []
    x_val, y_val = [], []

    path = "/home/ioan/Desktop/Database/"

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

    # old implementation (before dec. 4)
    # train_dataset = tf.data.Dataset.from_tensor_slices((np.asanyarray(x_train), np.asanyarray(y_train)))
    # val_dataset = tf.data.Dataset.from_tensor_slices((np.asanyarray(x_val), np.asanyarray(y_val)))
    # test_dataset = tf.data.Dataset.from_tensor_slices((np.asanyarray(x_test), np.asanyarray(y_test)))

    # 4 dec. update:
    # left away the old return implementation
    # because it looks easier to make
    # one_hot encoding and batch from this function
    # rather than breaking the already zipped dataset object in
    # data and label at the function call
    # Be aware of the one_hot and batch size!
    # The lambda expression can not be compiled!

    x_train = tf.data.Dataset.from_tensor_slices(np.asanyarray(x_train))
    y_train = keras.utils.to_categorical(y_train, 13)
    y_train = tf.data.Dataset.from_tensor_slices(np.asanyarray(y_train))

    x_val = tf.data.Dataset.from_tensor_slices(np.asanyarray(x_val))
    y_val = keras.utils.to_categorical(y_val, 13)
    y_val = tf.data.Dataset.from_tensor_slices(np.asanyarray(y_val))

    x_test = tf.data.Dataset.from_tensor_slices(np.asanyarray(x_test))
    y_test = keras.utils.to_categorical(y_test, 13)
    y_test = tf.data.Dataset.from_tensor_slices(np.asanyarray(y_test))

    return x_train, y_train, x_val, y_val, x_test, y_test



# extracts data from files of database
# located in directory: "/home/ioan/Desktop/Database/"
# makes a data augmentation by overlapping
# with a window of size 50%
# returns 6 tensorflow.data.Dataset objects:
# x_train, y_train, x_val, y_val, x_test, y_test
# Obs: val is the same with dev
def prepare_xy_train_val_test_BETA():
    # data base read and format:
    x_train, y_train = [], []
    x_test, y_test = [], []
    x_val, y_val = [], []

    path = "/home/ioan/Desktop/Database/"

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

    # old implementation (before dec. 4)
    # train_dataset = tf.data.Dataset.from_tensor_slices((np.asanyarray(x_train), np.asanyarray(y_train)))
    # val_dataset = tf.data.Dataset.from_tensor_slices((np.asanyarray(x_val), np.asanyarray(y_val)))
    # test_dataset = tf.data.Dataset.from_tensor_slices((np.asanyarray(x_test), np.asanyarray(y_test)))

    # 4 dec. update:
    # left away the old return implementation
    # because it looks easier to make
    # one_hot encoding and batch from this function
    # rather than breaking the already zipped dataset object in
    # data and label at the function call
    # Be aware of the one_hot and batch size!
    # The lambda expression can not be compiled!

    x_train = np.asanyarray(x_train)
    y_train = np.asanyarray(y_train)
    x_val = np.asanyarray(x_val)
    y_val = np.asanyarray(y_val)
    x_test = np.asanyarray(x_test)
    y_test = np.asanyarray(y_test)

    # reshape x_train
    x_train = np.reshape(x_train, (44030, 400, 1))
    print(x_train.shape)

    x_train = tf.data.Dataset.from_tensor_slices(x_train)
    y_train = keras.utils.to_categorical(y_train, 13)
    y_train = tf.data.Dataset.from_tensor_slices(y_train)

    x_val = tf.data.Dataset.from_tensor_slices(x_val)
    y_val = keras.utils.to_categorical(y_val, 13)
    y_val = tf.data.Dataset.from_tensor_slices(y_val)

    x_test = tf.data.Dataset.from_tensor_slices(x_test)
    y_test = keras.utils.to_categorical(y_test, 13)
    y_test = tf.data.Dataset.from_tensor_slices(y_test)

    return x_train, y_train, x_val, y_val, x_test, y_test

# NN architecture, basic TF API
# used dropout and first layer of batch normalization
# to effectively scale the input data
def nn_model(in_data):
    bn = tf.layers.batch_normalization(in_data)
    # 50 units, not necessary the best
    fc1 = tf.layers.dense(bn, 50)
    fc2 = tf.layers.dense(fc1, 50)
    fc2 = tf.layers.dropout(fc2)
    # output layer, 13 gestures
    fc3 = tf.layers.dense(fc2, 13)
    return fc3


# Dead code for test purpouse

# return x_train, y_train, x_test, y_test, x_val, y_val
# x_train = np.array(x_train)
# y_train = np.array(y_train)
# x_test = np.array(x_test)
# y_test = np.array(y_test)
# x_val = np.array(x_val)
# y_val = np.array(y_val)
#
# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)
# print(x_val.shape)
# print(y_val.shape)

# choose val dataset because is smaller
# for x, y in val_dataset:
#     print(x, y)

# ***************************************************************************
# extracts data from files of database
# located in directory: "/home/ioan/Desktop/Database/"
# makes a data augmentation by overlapping
# with a window of size 50%
# returns none, but creates csv files on Desktop
def write_database_toCSV():
    # data base read and format:
    x_train, y_train = [], []
    x_test, y_test = [], []
    x_val, y_val = [], []

    path = "/home/ioan/Desktop/Database/"

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
            label = (str(file))[5:7]
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

    file = open('/home/ioan/Desktop/x_train.csv', 'w+', newline='')
    # writing the data into the file
    with file:
        write = csv.writer(file)
        write.writerows(x_train)

    file = open('/home/ioan/Desktop/y_train.csv', 'w+', newline='')
    # writing the data into the file
    with file:
        write = csv.writer(file)
        write.writerows(y_train)

    file = open('/home/ioan/Desktop/x_test.csv', 'w+', newline='')
    # writing the data into the file
    with file:
        write = csv.writer(file)
        write.writerows(x_test)

    file = open('/home/ioan/Desktop/y_test.csv', 'w+', newline='')
    # writing the data into the file
    with file:
        write = csv.writer(file)
        write.writerows(y_test)

    return None


# extracts data from files of database
# located in directory: "/home/ioan/Desktop/Database/"
# makes a data augmentation by overlapping
# with a window of size 50%
# returns none, but creates text files on Desktop
def write_database_toCSV():
    # data base read and format:
    x_train, y_train = [], []
    x_test, y_test = [], []
    x_val, y_val = [], []

    path = "/home/ioan/Desktop/Database/"

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
            label = (str(file))[5:7]
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

    with open('/home/ioan/Desktop/x_train_txt', 'w') as outfile:
        for slice_2d in x_train:
            np.savetxt(outfile, slice_2d, fmt='%-7.0f')
            outfile.write('\n')  # to separate 2 arrays(for human readability scope)

    np.savetxt('/home/ioan/Desktop/y_train_txt', np.asanyarray(y_train))

    with open('/home/ioan/Desktop/x_test_txt', 'w') as outfile:
        for slice_2d in x_test:
            np.savetxt(outfile, slice_2d, fmt='%-7.0f')
            outfile.write('\n')

    np.savetxt('/home/ioan/Desktop/y_test_txt', np.asanyarray(y_test))

    # simple np.savetxt() won't work for x files since
    # we are dealing with 3D arrays. Next line will give an error
    # np.savetxt('/home/ioan/Desktop/x_train_txt', np.asanyarray(x_train))

    return None
