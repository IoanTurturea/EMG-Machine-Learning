from test import *
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# READ ME:
# solve the iterator problem
# but I do know the concepts behind tf API
# for iterator
# that why I prefered read from CSV method
# but we have that "cardinality error" - see main3.py
# must solve iterator problem to move on
# with this implementation

epochs = 10

# import already prepared database
# retuns 3 tf.data.Dataset objects
train_dataset, val_dataset, test_dataset = prepare_database()

# create general iterator
# iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
iterator = train_dataset
next_element = iterator.get_next()

# make datasets that we can initialize separately,
# but using the same structure via the common iterator
training_init_op = iterator.make_initializer(train_dataset)
validation_init_op = iterator.make_initializer(val_dataset)
test_init_op = iterator.make_initializer(test_dataset)


# create the neural network model
logits = nn_model(next_element[0])


# add the optimizer and loss
loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=next_element[1], logits=logits))
optimizer = tf.train.AdamOptimizer().minimize(loss)
# get accuracy
prediction = tf.argmax(logits, 1)
equality = tf.equal(prediction, tf.argmax(next_element[1], 1))
accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
init_op = tf.global_variables_initializer()


# run the training
with tf.Session() as sess:
    sess.run(init_op)
    sess.run(training_init_op)
    for i in range(epochs):
        l, _, acc = sess.run([loss, optimizer, accuracy])
        if i % 50 == 0:
            print("Epoch: {}, loss: {:.3f}, training accuracy: {:.2f}%".format(i, l, acc * 100))
    # now setup the validation run
    valid_iters = 100
    # re-initialize the iterator, but this time with validation data
    sess.run(validation_init_op)
    avg_acc = 0
    for i in range(valid_iters):
        acc = sess.run([accuracy])
        avg_acc += acc[0]
    print("Average validation set accuracy over {} iterations is {:.2f}%"
          .format(valid_iters, (avg_acc / valid_iters) * 100))




# # NN architecture:
# classifier = Sequential()
#
# classifier.add(Dense(units=16, activation='relu', input_dim=30))
# classifier.add(Dense(units=8, activation='relu'))
# classifier.add(Dense(units=6, activation='relu'))
# classifier.add(Dense(units=1, activation='sigmoid'))
#
# # Optimizer and Loss function:
# classifier.compile(optimizer='adam', loss='binary_crossentropy')
#
# # Fitting:
# classifier.fit(x_train, y_train, batch_size=1, epochs=50)
# print(classifier.predict(x_train).round())