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


# extracts features and returns 6 numpy arrays
def extract_features_from_channel():
    # data base read and format:
    x_train, y_train = [], []
    x_test, y_test = [], []
    x_val, y_val = [], []

    # window size, choose it to be 50 samples
    # because 50 & 200Hz(sampling rate) = 250ms
    # a sample = one line from a (*,8) matrix
    N = 50
    overlap_procent = 0.5 # input it in range [0,1]
    overlap = int(N * overlap_procent)

    folders = os.listdir(path)  # 3 folders in Database: Train, Val, Test

    # iterate in folders
    for folder in folders:
        data = os.path.join(path, folder)
        files = os.listdir(data)

        # iterate files in a folder
        for file in files:
            filepath = data + "/" + str(file)
            signal2 = np.loadtxt(filepath) # renamed it 'signal2'. because in a library exists 'signal' function
            label = int((str(file))[5:7])

            step = N - overlap
            real_max_value = len(signal2) - overlap
            max_value = int(real_max_value / step) * step  # multiples of overlap must fit signal length.

            # Here was a minor bug: windows dimensions not equal to last window dimension for overlap != 25
            for i in range(0, max_value, step):
                window = signal2[i: (i + N)]
                features = []

                # calculate features for each channel and append them
                for nr_of_channels in range(0, window.shape[1]):
                    channel = window[:, nr_of_channels]
                    channel = np.reshape(channel, (1, N))

                    # calculate features:

                    # time descriptors
                    MAV = (1 / len(channel)) * abs_sum(channel)
                    SSC_positions = np.nonzero(np.diff(channel > 0))[0]
                    SSC = SSC_positions.size / channel.size  # returns frequnecy of SSC, not sure this is correct
                    ZCR = SSC_positions.size
                    WL = waveform_length(channel)
                    Skewness = skew(channel) # normal da 0, anormal este != 0
                    RMS = np.sqrt(np.mean(channel ** 2))
                    # Hjorth = ?
                    IEMG = integratedEMG(channel)
                    # Autoregression = ?
                    # SampEn = sampen.sampen2(channel) # vezi: https://pypi.org/project/sampen/
                    # EMGHist = ?

                    # frequency descriptors
                    powerspectrum = np.abs(np.fft.fft(channel)) ** 2
                    powerspectrum[np.where(powerspectrum == 0)] = 10**-10 # pt ca: RuntimeWarning: divide by zero encountered in log
                    Cepstral = np.fft.ifft(np.log(powerspectrum)) # vezi: https://dsp.stackexchange.com/questions/48886/formula-to-calculate-cepstral-coefficients-not-mfcc
                    # mDWT = nu a mers import pywt(oricum e doar DWT, deci fara 'marginal')
                    # vezi: https://pywavelets.readthedocs.io/en/latest/install.html
                    f, t, Zxx = signal.stft(channel, fs=200, nperseg=len(channel))

                    #make mean where is an array:
                    mav =  np.mean(MAV)
                    skewness = np.mean(Skewness)
                    # sampen = np.mean(SampEn)
                    cepstral = np.mean(np.abs(Cepstral))
                    zxx = np.mean(np.abs(Zxx))

                    features.append(np.array([mav, SSC, ZCR, WL, RMS, IEMG, cepstral, zxx], dtype = float))
                    # obs: Skewness era mereu 0(ceea ce e bine), asa ca l-am scos ca nu oferea informatii si ca sa am 8 descriptori

                features = np.asanyarray(features)
                features = np.reshape(features, (window.shape[1] * features[0].size))
                if folder == 'Train':
                    x_train.append(features)
                    y_train.append(label)
                elif folder == 'Test':
                    x_test.append(features)
                    y_test.append(label)
                elif folder == 'Val':
                    x_val.append(features)
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


# helper functions:
def abs_sum(arr):
    suma = 0
    for i in arr:
        suma += i
    return suma


def waveform_length(arr):
    suma = 0
    arr = arr.reshape(arr.size) # make it flat
    for i in range(1, arr.size):
        suma += abs(arr[i] - arr[i - 1])
    return suma


def integratedEMG(arr):
    suma = 0
    arr = arr.reshape(arr.size)  # make it flat
    for i in range(1, arr.size):
        suma += abs(arr[i])
    return suma

