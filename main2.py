from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
from functions import *

# READ ME:(5 dec.)
# for unknown reason(as already happened in october)
# the csv file although have the same dimensions
# shape attribute un PyCharm have slightly
# different values, and error appears:
# ValueError: Data cardinality is ambiguous:

# READ ME(12 dec.)
# I tried many solvings, but could not make it out...
# so I think it is better to leave this implementation
# since it has some particularities, such as
# a cell in the csv is an array. So lets continue with tf.dataset

# call this method only when you want to recreate the csv files
# else skip, because it is time penalty, more than 10 min.
# prepare_database_toCSV()
# or txt file version
prepare_database_totxt()

# read from CSV (method one)
# xtrain = pd.read_csv(r'/home/ioan/Desktop/x_train.csv')
# ytrain = pd.read_csv(r'/home/ioan/Desktop/y_train.csv')
# xtest = pd.read_csv(r'/home/ioan/Desktop/x_test.csv')
# ytest = pd.read_csv(r'/home/ioan/Desktop/y_test.csv')
#
# x_train = np.array(xtrain)
# y_train = np.array(ytrain)
# x_test = np.array(xtest)
# y_test = np.array(ytest)

# OR: read from txt (method two)
x_train = np.loadtxt('/home/ioan/Desktop/x_train_txt')
y_train = np.loadtxt('/home/ioan/Desktop/y_train_txt')
x_test = np.loadtxt('/home/ioan/Desktop/x_test_txt')
y_test = np.loadtxt('/home/ioan/Desktop/y_test_txt')

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# NN architecture:
classifier = Sequential()

classifier.add(Dense(units= 16, activation = 'relu', input_dim = 50))
classifier.add(Dense(units = 8, activation = 'relu'))
classifier.add(Dense(units = 6, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

classifier.summary()

# Optimizer and Loss function:
classifier.compile(optimizer='adam', loss = 'binary_crossentropy')

# Fitting:
classifier.fit(x_train, y_train, batch_size = 8, epochs=10)
print(classifier.predict(x_train).round())

# Testare:
# y_pred = classifier.predict(x_test)
# y_pred = [ 1 if y>=0.5 else 0 for y in y_pred]
#
#
# # Acuratete:
# total = 0
# correct = 0
# wrong = 0
# for i in range(len(y_pred)):
#     total += 1
#     if(y_test.at[i,0] == y_pred[i]):
#         correct += 1
#     else:
#         wrong += 1
#
# print("Total: " + str(total))
# print("Correct: " + str(correct))
# print("Wrong: " + str(wrong))