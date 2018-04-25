#!/usr/bin/env python
# encoding: utf-8
'''
Inception模型
'''

import keras
from keras.layers import Conv2D, MaxPool2D, Input

input_img = Input(shape=(256, 256, 3))

tower1 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_img)
tower1 = Conv2D(64, (3, 3), padding='same', activation='relu')(tower1)

tower2 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_img)
tower2 = Conv2D(64, (5, 5), padding='same', activation='relu')(tower2)

tower3 = MaxPool2D((3, 3), strides=(1,1), padding='same')
tower3 = Conv2D(64, (1, 1), padding='same', activation='relu')(tower3)

output = keras.layers.concatenate([tower1, tower2, tower3], axis=1)