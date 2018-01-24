import numpy as np
import dataset as dset
import resnet
import pickle
from keras import Sequential
from keras.layers import Dense

def resnet152(datasetPath, modelPath, historyPath, batchSize = 64, epochs = 1000):

    [samples, configs, validationSamples, validationConfigs, testSamples, testConfigs, itemLen, configLen] = pickle.load(open(datasetPath, 'rb'))

    print('Samples count: {0}'.format(len(samples)))
    print('Validation samples count: {0}'.format(len(validationSamples)))
    print('Test samples count: {0}'.format(len(testSamples)))
    print('Config max len: {0}'.format(configLen))

    resnetBuilder = resnet.ResnetBuilder()
    model = resnetBuilder.build_resnet_152([1, itemLen, 8], configLen)
    model.compile(loss='mean_squared_error', optimizer='rmsprop')

    history = model.fit(samples.reshape([len(samples), itemLen, 8, 1]), 
                    configs,
                    batch_size=batchSize,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(validationSamples.reshape([len(validationSamples), itemLen, 8, 1]), validationConfigs))
 
    score = model.evaluate(validationSamples, validationConfigs, verbose=0)
    print('Loss on validation set:', score)

    model.save(modelPath)
    pickle.dump(history, open(historyPath, 'wb'))

    return

def sequentialDense(datasetPath, modelPath, historyPath, batchSize = 64, epochs = 1000):

    [samples, configs, validationSamples, validationConfigs, testSamples, testConfigs, itemLen, configLen] = pickle.load(open(datasetPath, 'rb'))

    print('Samples count: {0}'.format(len(samples)))
    print('Validation samples count: {0}'.format(len(validationSamples)))
    print('Test samples count: {0}'.format(len(testSamples)))
    print('Config max len: {0}'.format(configLen))

    model = Sequential()
    model.add(Dense(256, activation='relu', input_shape=[itemLen* 8]))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(22, activation='linear'))

    model.compile(loss='mean_squared_error', optimizer='rmsprop')

    history = model.fit(samples, 
                    configs,
                    batch_size=batchSize,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(validationSamples, validationConfigs))
 
    score = model.evaluate(validationSamples, validationConfigs, verbose=0)
    print('Loss on validation set:', score)

    model.save(modelPath)
    pickle.dump(history, open(historyPath, 'wb'))

    return