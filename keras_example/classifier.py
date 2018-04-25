#!/usr/bin/env python
# encoding: utf-8
import numpy as np
np.random.seed(1314)
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop

# X shape(60000, 28, 28) y shape (60000, )
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape)
print(y_train.shape)
X_train = X_train.reshape(X_train.shape[0], -1) / 255
X_test = X_test.reshape(X_test.shape[0], -1) / 255
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

# build your neural net
model = Sequential([
    Dense(32, input_dim=784, activation='relu'),
    Dense(10, activation='softmax')
])

# define your optimizer
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-8, decay=0.0)

# add metrics to get more results your want to see
model.compile(optimizer=rmsprop,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print('Training.............')
# trian the model
model.fit(X_train, y_train, epochs=2, batch_size=32)


print('\nTesting.........')
loss, accuracy = model.evaluate(X_test, y_test)
print('test loss:', loss)
print('test accuracy:', accuracy)