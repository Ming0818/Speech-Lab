import numpy as np
import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation

if __name__== "__main__":
    # import data
    train_data  = np.load('train.npz')
    val_data  = np.load('val.npz')
    test_data  = np.load('test.npz')

    lmfcc_train_x = train_data['lmfcc'].astype('float32')
    mspec_train_x = train_data['mspec'].astype('float32')
    train_y = train_data['targets'].astype('float32')

    lmfcc_val_x = val_data['lmfcc'].astype('float32')
    mspec_val_x = val_data['mspec'].astype('float32')
    val_y = val_data['targets'].astype('float32')

    lmfcc_test_x = test_data['lmfcc'].astype('float32')
    mspec_test_x = test_data['mspec'].astype('float32')
    test_y = test_data['targets'].astype('float32')

    # get the StateList
    phoneHMMs = np.load('lab2_models.npz')['phoneHMMs'].item()
    phones = sorted(phoneHMMs.keys())
    nstates = {phone: phoneHMMs[phone]['means'].shape[0] for phone in phones}
    stateList = [ph + '_' + str(id) for ph in phones for id in range(nstates[ph])]
    output_dim = len(stateList)
    train_y = np_utils.to_categorical(train_y, output_dim)
    val_y = np_utils.to_categorical(val_y, output_dim)
    test_y = np_utils.to_categorical(test_y, output_dim)

    # define model
    model = Sequential()
    # Stacking layers
    model.add(Dense(units=256, activation='relu', input_dim=lmfcc_train_x.shape[1]))
    model.add(Dense(units=256, activation='softmax'))

    # configure its learning process
    model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])

    model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))

    # iterate on your training data in batches:
    # x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
    model.fit(lmfcc_train_x, train_y, epochs=5, batch_size=256)

    # Evaluate your performance
    loss_and_metrics = model.evaluate(lmfcc_val_x, val_y, batch_size=256)

    # generate predictions on new data
    classes = model.predict(lmfcc_test_x, batch_size=256)
