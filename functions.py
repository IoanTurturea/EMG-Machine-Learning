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


# returns 6 np arrays:
# x_train, y_train, x_val, y_val, x_test, y_test
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


# extracts features and returns 6 numpy arrays
def prepare_xy_train_val_test_features():
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
    # hamming = np.hamming(N * 8)  # did not improve accuracy :(
    # hamming = np.reshape(hamming, (N*8, )) # reshape both window if wanted, but does not make improvements

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
                # window = np.reshape(window, (N * 8))
                # window = np.reshape(window, (N*8, )) # reshape both hamming if wanted, but does not make improvements
                # window *= hamming
                # rehsape it (1, 400)
                window = np.reshape(window, (1, N*8))

                # calculate features:
                
                # time descriptors
                MAV = (1 / len(window)) * abs_sum(window)
                #SSC_positions = np.where(np.diff(np.sign(window)))[0]  # position is optional but might use it later
                SSC_positions = np.nonzero(np.diff(window > 0))[0]
                SSC = SSC_positions.size / window.size  # returns frequnecy of SSC, not sure this is correct
                ZCR = SSC_positions.size
                WL = waveform_length(window)
                Skewness = skew(window) # normal da 0, anormal este != 0
                RMS = np.sqrt(np.mean(window ** 2))
                # Hjorth = ?
                IEMG = integratedEMG(window)
                # Autoregression = ?
                # SampEn = sampen.sampen2(window) # vezi: https://pypi.org/project/sampen/
                # EMGHist = ?
                a = 1

                # frequency descriptors
                # powerspectrum = np.abs(np.fft.fft(window)) ** 2
                # Cepstral = np.fft.ifft(np.log(powerspectrum)) # vezi: https://dsp.stackexchange.com/questions/48886/formula-to-calculate-cepstral-coefficients-not-mfcc
                # mDWT = nu a mers import pywt(oricum e doar DWT, deci fara 'marginal')
                # vezi: https://pywavelets.readthedocs.io/en/latest/install.html
                # f, t, Zxx = signal.stft(window, fs=200, nperseg=len(window))
                """
                - Asa arata dimensiunile fiecarui feature:
                MAV = ndarray: (8:)
                SSC = float
                ZCR = int64
                WL = ndarray: (8:)
                Skewness = ndarray: (8:)
                RMS = float64
                IEMG = ndarray: (8:)
                SampEn = ndarray: (10:)
                Cepstral = ndarray: (10:8) - numere imaginare
                Zxx = ndarray: (10,5,3)
                - fiindca tipurile sunt diferite am facut media celor care sunt array
                  ca sa fie toti de tipul float
                - obs: elementele care au dim: (8:0) aplica operatia aceea pe lungimea N a ferestrei.
                  Nu stiu data trebuie aplicata pe lungimea N, sau pe nr. de coloane = 8.
                """

                #make mean where is an array:
                mav =  np.mean(MAV)
                wl = np.mean(WL)
                skewness = np.mean(Skewness)
                iemg = np.mean(IEMG)
                #sampen = np.mean(SampEn)
                #cepstral = np.abs(np.mean(Cepstral))
                #zxx = np.mean(Zxx)

                #features = namedtuple(MAV, SSC, ZCR, WL, Skewness, RMS, IEMG, SampEn, Cepstral, Zxx)
                #features = np.array([MAV, SSC, ZCR, WL, Skewness, RMS, IEMG, SampEn, Cepstral, Zxx], dtype=object)
                features = np.array([mav, SSC, ZCR, wl, skewness, RMS, iemg], dtype = float)

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

