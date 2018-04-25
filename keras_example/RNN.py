#!/usr/bin/env python
# encoding: utf-8
import numpy as np
np.random.seed(1314)

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import SimpleRNN, Activation, Dense
from keras.optimizers import Adam
from keras.utils import np_utils

TIME_STEPS = 28
INPUT_SIZE = 28
BATCH_SIZE = 50
BATCH_INDEX = 0
OUTPUT_SIZE = 10
CELL_SIZE = 50
LR = 0.001

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28, 28) / 255
X_test = X_test.reshape(-1, 28, 28) / 255
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

# build RNN model
model = Sequential()
model.add(SimpleRNN(
    units=CELL_SIZE,
    batch_input_shape=(BATCH_SIZE, TIME_STEPS, INPUT_SIZE)
))
model.add(Dense(10, activation='softmax'))

adam = Adam(lr=LR)

model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

# training
for step in range(4001):
    # data shape (batch_num, steps, inputs/outputs)
    X_batch = X_train[BATCH_INDEX: BATCH_INDEX + BATCH_SIZE, :, :]
    Y_batch = y_train[BATCH_INDEX:BATCH_INDEX + BATCH_SIZE, :]
    cost = model.train_on_batch(X_batch, Y_batch)

    BATCH_INDEX += BATCH_SIZE
    BATCH_INDEX = 0 if BATCH_INDEX >= X_train.shape[0] else BATCH_INDEX

    if step % 500 == 0:
        cost, accuracy = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=False)
        print('test cost:', cost, 'test accuracy:', accuracy)

