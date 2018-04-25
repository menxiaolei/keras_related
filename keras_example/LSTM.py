#!/usr/bin/env python
# encoding: utf-8
import numpy as np
np.random.seed(1314)

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import LSTM, Activation, Dense, TimeDistributed
from keras.optimizers import Adam
from keras.utils import np_utils
import matplotlib.pyplot as plt

BATCH_START = 0
TIME_STEPS = 20
BATCH_SIZE = 50
INPUT_SIZE = 1
OUTPUT_SIZE = 1
CELL_SIZE = 20
LR = 0.006


def get_batch():
    global BATCH_START, TIME_STEPS
    # xs shape (50batch, 20steps)
    xs = np.arange(BATCH_START, BATCH_START+TIME_STEPS*BATCH_SIZE).reshape((BATCH_SIZE, TIME_STEPS)) / (10*np.pi)
    seq = np.sin(xs)
    res = np.cos(xs)
    BATCH_START += TIME_STEPS
    # plt.plot(xs[0, :], res[0, :], 'r', xs[0, :], seq[0, :], 'b--')
    # plt.show()
    return [seq[:, :, np.newaxis], res[:, :, np.newaxis], xs]

# model
model = Sequential()
model.add(LSTM(units=CELL_SIZE,
               batch_input_shape=(BATCH_SIZE, TIME_STEPS, INPUT_SIZE),
               return_sequences=True,   # 在每个时间步上都返回值，即在这里每个时间序列输出28个值
               stateful=True,  # 前一个序列的最后状态和后一个序列是否有关联，在这里有关联
               ))
model.add(TimeDistributed(Dense(OUTPUT_SIZE)))    # 因为上一层Lstm中return_sequences=True，所以要加Timedistribated

adam = Adam(lr=LR)
model.compile(optimizer=adam, loss='mse')

print('Training ------------')
for step in range(1001):
    # data shape = (batch_num, steps, inputs/outputs)
    X_batch, Y_batch, xs = get_batch()
    cost = model.train_on_batch(X_batch, Y_batch)
    pred = model.predict(X_batch, BATCH_SIZE)
    plt.plot(xs[0, :], Y_batch[0].flatten(), 'r', xs[0, :], pred.flatten()[:TIME_STEPS], 'b--')
    plt.ylim((-1.2, 1.2))
    plt.draw()
    plt.pause(0.1)
    if step % 10 == 0:
        print('train cost: ', cost)



