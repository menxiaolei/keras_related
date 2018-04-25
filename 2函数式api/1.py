#!/usr/bin/env python
# encoding: utf-8

# 定义全连接
import keras
from keras.models import Model
from keras.layers import Input, Dense

# 生成虚拟数据
import numpy as np
x_train = np.random.random((10000, 20))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(10000,1)), num_classes=10)
x_test = np.random.random((100, 20))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(100,1)), num_classes=10)

# 这部分返回一个张量
inputs = Input(shape=(20,))

# 层的实例都是可调的，它以张量为参数，并且返回一个张量
x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

# 这部分创建了一个包含输入层和3个全连接层的模型
model = Model(inputs=inputs, outputs=predictions)
# 打印模型概述信息
model.summary()
# 返回包含模型配置信息的字典，可以根据配置信息重新实例化模型
config = model.get_config()
print(config)
model_copy = Model.from_config(config)
model_copy.summary()

# print(model_copy.get_weights())

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
# 开始训练
hist = model.fit(x_train, y_train, epochs=20, batch_size=128, validation_split=0.2)

# 在每个epoch后记录训练集和验证集的误差和准确率,返回一个字典
print(hist.history)
#
predicts = model.predict(x_test, batch_size=128)
print(predicts)
print(predicts.shape)