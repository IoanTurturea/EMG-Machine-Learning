import numpy as np
import h5py
from tensorflow import keras
from tensorflow.python.keras.callbacks import ModelCheckpoint
import pandas as pd
from functions import *
import matplotlib.pyplot as plt
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Dropout, Dense
from tensorflow.python.keras.models import Model, load_model, save_model

print("TF version:", tf.__version__)

x_train, y_train, x_dev, y_dev, x_test, y_test = prepare_xy_train_val_test_asnumpyarrays()  # returns numpy arrays
#train_dataset = tf.data.Dataset.zip((x_train, y_train))
#train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

input = Input((400,), name='input')
hidden = Dense(64, activation='relu', name='hidden_1')(input)
dropout = Dropout(0.2, name='dropout_1')(hidden)
hidden = Dense(32, activation='relu', name='hidden_2')(dropout)
dropout = Dropout(0.2, name='dropout_2')(hidden)
hidden = Dense(32, activation='relu', name='hidden_3')(dropout)
dropout = Dropout(0.2, name='dropout_3')(hidden)
output = Dense(13, activation='softmax', name='output')(dropout)
model = Model(inputs=input, outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model = load_model('1.h5')

accuracy = model.fit(x = x_train, y = y_train, callbacks=[ModelCheckpoint('1.h5', save_best_only=True, verbose=1)],
                     batch_size=64, epochs = 2048, verbose=2, validation_data=(x_dev, y_dev))

model.summary()


# Results:
results = model.evaluate(x_test, y_test, batch_size = 64)
print(f'\nTest loss: {results[0]} \n Test accuracy: {results[1]}\n')

# plot loss for train and test
plt.figure()
loss_curve = accuracy.history["loss"]
loss_curve_dev = accuracy.history["val_loss"]

plt.subplot(211)
plt.plot(loss_curve, label= "Train")
plt.legend(loc='upper left')
plt.title("Loss train")

plt.subplot(212)
plt.plot(loss_curve_dev, label= "Val")
plt.legend(loc='upper left')
plt.title("Loss val")

#plt.show()
plt.savefig("loss.jpg")

