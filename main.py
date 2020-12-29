import numpy as np
import h5py
from tensorflow import keras
from tensorflow.python.keras.callbacks import ModelCheckpoint
import pandas as pd
from functions import *
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Dropout, Dense
from tensorflow.python.keras.models import Model, load_model, save_model

print("TF version:", tf.__version__)

x_train, y_train, x_dev, y_dev, _, _ = prepare_xy_train_val_test()
#train_dataset = tf.data.Dataset.zip((x_train, y_train))

input = Input((50, 8), name='input')
hidden = Dense(64, activation='relu', name='hidden_1')(input)
dropout = Dropout(0.2, name='dropout_1')(hidden)
hidden = Dense(32, activation='relu', name='hidden_2')(dropout)
dropout = Dropout(0.2, name='dropout_2')(hidden)
hidden = Dense(32, activation='relu', name='hidden_3')(dropout)
dropout = Dropout(0.2, name='dropout_3')(hidden)
output = Dense(7, activation='softmax', name='output')(dropout)
model = Model(inputs=input, outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# model = load_model('1.h5')
# y_train = keras.utils.to_categorical(y_train, 13)
# y_dev = keras.utils.to_categorical(y_dev, 13)

accuracy = model.fit(x_train, callbacks=[ModelCheckpoint('1.h5', save_best_only=True, verbose=1)],
                     batch_size=64, epochs=2048, verbose=2, validation_data=(x_dev, y_dev))

model.summary()


# model.fit(train_dataset,
#             batch_size=64,
#             epochs=2048,
#             verbose=1,
#             validation_data=(x_dev, y_dev))
# print(accuracy)
# model.save('firstmodel256_128b.h5')
# model.save('1.h5')
# save_model(model, '1', saveformat='h5')
# y_predict = model.predict(x_dev)
# label_predict = np.argmax(y_predict, axis=1)

