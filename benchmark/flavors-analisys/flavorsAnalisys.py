import benchmarks as fb
#import keras
#from keras.models import Sequential
#from keras.layers import Dense, Dropout
import os
import json 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import resnet
import pickle
#import dataset as d

#using only first device
os.environ["CUDA_VISIBLE_DEVICES"]="0"

dataPath = 'D:\\Projekty\\flavors-results\\P100-keys-top10\\keysResults.zip'
dataInfoPath = 'D:\\Projekty\\flavors-results\\P100-keys-top10\\dataInfo'

#findRandomDatasetPath = 'D:\\Projekty\\flavors-results\\P100-keys-top145\\findRandomDataset_shortConfigs.pkl'
#buildDatasetPath = 'D:\\Projekty\\flavors-results\\P100-keys-top10\\buildDataset.pkl'

#rawData = fb.throughputs(pd.read_csv(dataPath4, sep=';'))
#fb.bestConfigs(rawData, dataInfoPath4)

#validationShare = 0.2
#testShare = 0.2

#data = d.load(dataInfoPath)
#buildDataset = d.build(data, validationShare, testShare, 'bestBuildConfig')
#findRandomDataset = d.build(data, validationShare, testShare, 'bestFindRandomConfig')

#[samples, configs, validationSamples, validationConfigs, testSamples, testConfigs, itemLen] = pickle.load(open(findRandomDatasetPath, 'rb'))

#pickle.dump(buildDataset, open(buildDatasetPath, 'wb'))
#pickle.dump(findRandomDataset, open(findRandomDatasetPath, 'wb'))

#print('Samples count: {0}'.format(len(samples)))
#print('Validation samples count: {0}'.format(len(validationSamples)))
#print('Test samples count: {0}'.format(len(testSamples)))
#print('Class count: {0}'.format(classCount))

#rawData = pd.read_csv(dataPath, sep = ';', compression = 'zip')
#rawData = fb.throughputs(pd.read_csv('D:\\Projekty\\flavors-results\\P100-keys-top10-randomCount\\keysResults.csv', sep = ';'))

#fb.buildThroughput(rawData)
#fb.findThroughput(rawData)
#fb.findRandomThroughput(rawData)

fb.buildConfigHist(dataInfoPath)
plt.show()
fb.findRandomConfigHist(dataInfoPath)
plt.show()
fb.findConfigHist(dataInfoPath)
plt.show()

#resnetBuilder = resnet.ResnetBuilder()
#model = resnetBuilder.build_resnet_50([1, itemLen, 8], itemLen)

#batch_size = 64
#epochs = 5

#sgd = keras.optimizers.SGD(lr=0.3, momentum=0.01, decay=0.01, nesterov=False)
#model.compile(loss='mean_squared_error', optimizer='rmsprop')

#history = model.fit(samples.reshape([len(samples), itemLen, 8, 1]), 
#                    configs,
#                    batch_size=batch_size,
#                    epochs=epochs,
#                    verbose=1,
#                    validation_data=(validationSamples.reshape([len(validationSamples), itemLen, 8, 1]), validationConfigs))
 
#score = model.evaluate(validationSamples, validationConfigs, verbose=0)
#print('Test loss on validation set:', score)
#model.save('D:\\Projekty\\flavors-results\\P100-keys-top10\\resnet50_100epochs_adam.h5')

#samples = samples.reshape(len(samples), itemLen * 8)
#validationSamples = validationSamples.reshape(len(validationSamples), itemLen * 8)

#model = Sequential()
#model.add(Dense(256, activation='relu', input_shape=[itemLen* 8]))
#model.add(Dense(256, activation='relu'))
#model.add(Dense(512, activation='relu'))
#model.add(Dense(512, activation='relu'))
#model.add(Dense(1024, activation='relu'))
#model.add(Dense(2048, activation='relu'))
#model.add(Dense(4096, activation='relu'))
#model.add(Dense(4096, activation='relu'))
#model.add(Dense(4096, activation='relu'))
#model.add(Dense(4096, activation='relu'))
#model.add(Dense(4096, activation='relu'))
#model.add(Dense(4096, activation='relu'))
#model.add(Dense(2048, activation='relu'))
#model.add(Dense(2048, activation='relu'))
#model.add(Dense(2048, activation='relu'))
#model.add(Dense(2048, activation='relu'))
#model.add(Dense(2048, activation='relu'))
#model.add(Dense(2048, activation='relu'))
#model.add(Dense(1024, activation='relu'))
#model.add(Dense(1024, activation='relu'))
#model.add(Dense(512, activation='relu'))
#model.add(Dense(512, activation='relu'))
#model.add(Dense(256, activation='relu'))
#model.add(Dense(256, activation='relu'))
#model.add(Dense(128, activation='relu'))
#model.add(Dense(128, activation='relu'))
#model.add(Dense(64, activation='relu'))
#model.add(Dense(64, activation='relu'))
#model.add(Dense(32, activation='relu'))
#model.add(Dense(22, activation='linear'))

#model = Sequential()
#model.add(Dense(itemLen* 8,init='uniform', activation='linear', input_shape=[itemLen* 8]))
#model.add(Dense(itemLen* 8,init='uniform', activation='linear'))
#model.add(Dense(itemLen* 8,init='uniform', activation='linear'))
#model.add(Dense(itemLen,init='uniform', activation='linear'))
 
#model.summary()
 
#model.compile(loss='mean_squared_error',
#              optimizer='sgd')

#history = model.fit(samples, 
#                    configs,
#                    batch_size=batch_size,
#                    epochs=epochs,
#                    verbose=1,
#                    validation_data=(validationSamples, validationConfigs))

