'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Flatten
from keras.optimizers import RMSprop
from keras import optimizers
import numpy as np
import random

from keras.optimizers import SGD

from keras.layers import Embedding

import keras.backend.tensorflow_backend as KTF

random.seed(100)

stateList = np.load('stateList.npz')['stateList']
num_classes = stateList.shape[0]

# data =np.load('data_stan_nody/traindata_stan.npz',encoding='bytes')
data =np.load('data_stan_nody/traindata_stan.npz')
x_train = data['mspec']
y_train =  data['targets']

data =np.load('data_stan_nody/valdata_stan.npz')
x_val = data['mspec']
y_val =  data['targets']

data =np.load('data_stan_nody/testdata_stan.npz')
x_test = data['mspec']
y_test =  data['targets']

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_val = x_val.astype('float32')
y_test = y_test.astype('float32')
y_train = y_train.astype('float32')
y_val = y_val.astype('float32')

# convert class vectors to binary class matrices
# one hot encoding
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)


# build a model
model = Sequential()

input_d = x_train.shape[1]

# # three hidden layer MLP
model.add(Dense(256, activation='relu', input_shape=(input_d,)))
model.add(Dropout(0.1))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.1))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.1))

model.add(Dense(num_classes, init = "zero", activation='softmax', input_shape=(256,)))

model.summary()

# sgd = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
adam = keras.optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

# RMSprop()
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

# training
model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val), batch_size=256)

prediciton = model.predict(x_test)
np.savez("prediction_mspec.npz",prediciton = prediciton)

# evaluation
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

