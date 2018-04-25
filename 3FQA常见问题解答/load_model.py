#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import keras
from keras.models import load_model

x_test = np.random.random((100, 20))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(100,1)), num_classes=10)

model = load_model('my_model.h5')

model.evaluate(x_test, y_test, batch_size=128)