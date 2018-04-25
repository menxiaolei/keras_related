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
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
# 开始训练
model.fit(x_train, y_train, epochs=20, batch_size=128)

# 保存模型及权重
# 也可通过save_weights只保存权重(load_weights)，也可通过model.to_json(model_from_json)
model.save('my_model.h5')

score = model.evaluate(x_test, y_test, batch_size=128)
print(score)