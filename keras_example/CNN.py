#!/usr/bin/env python
# encoding: utf-8
import tensorflow as tf

import numpy as np
np.random.seed(1314)
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.optimizers import Adam

with tf.device('/gpu:0'):
    # X shape(60000, 28, 28) y shape (60000, )
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    print(X_train.shape)
    print(y_train.shape)
    X_train = X_train.reshape(-1, 1, 28, 28)
    X_test = X_test.reshape(-1, 1, 28, 28)
    y_train = np_utils.to_categorical(y_train, num_classes=10)
    y_test = np_utils.to_categorical(y_test, num_classes=10)

    # build your neural net
    model = Sequential()
    # (1,28,28) ---> (channel, height, width)
    model.add(Conv2D(filters=32, kernel_size=(5,5), padding='same', input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2,2), padding='same'))
    model.add(Conv2D(filters=64, kernel_size=(5,5), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), padding='same'))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    # define your optimizer
    adam = Adam(lr=1e-4)

    # add metrics to get more results your want to see
    model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    print('Training.............')
    # trian the model
    hist = model.fit(X_train, y_train, epochs=5, batch_size=32)

    print('\nTesting.........')
    loss, accuracy = model.evaluate(X_test, y_test)
    print('test loss:', loss)
    print('test accuracy:', accuracy)

    print(hist.history)