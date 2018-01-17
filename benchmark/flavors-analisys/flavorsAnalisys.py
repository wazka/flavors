import benchmarks as fb
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
import os
import json 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import resnet
import pickle

#using only first device
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def load(paths):
    data = []
    if type(paths) is str:
        return loadFromFile(paths)
    else:
        for path in paths:
            data.extend(loadFromFile(path))
        return data

def loadFromFile(path):
    data = []
    for fileName in os.listdir(path):
        with open(os.path.join(path, fileName)) as file:
            data.append(json.load(file))

    return data

def getConfig(data, param):
    return ''.join(map(str, data[param]))

def appendOptional(stats, pos, param):
    if pos[param] == None:
        stats.append(0.0)
    else:
        stats.append(pos[param])

def getStats(data):
    stats = []
    for pos in data['dataInfo']:
        stats.append(float(pos['mean']))
        stats.append(float(pos['max']))
        stats.append(float(pos['min']))
        stats.append(float(pos['std']))
        stats.append(float(pos['n']))
        appendOptional(stats, pos, 'kurtosis')
        appendOptional(stats, pos, 'skewness')
        appendOptional(stats, pos, 'variance')
    return stats

def getLabels(data, param):
    labels = dict()

    for info in data:
        config = getConfig(info, param)

        if not config in labels:
            labels[config] = len(labels)

    return labels

def setsImpl(data, labels, classCount, itemLen, param):
    stats = []
    configs = []

    for info in data:
        config = getConfig(info,param)
        configs.append(labels[config])
        stats.extend(getStats(info))

    samples = np.array(stats).reshape(len(configs), itemLen, 8, 1)
    configs = keras.utils.to_categorical(np.array(configs), classCount)

    return samples, configs

def dataset(data, validationShare, testShare, param):
    labels = getLabels(data, param)
    classCount = labels[max(labels, key=lambda config: labels[config])] + 1
    maxSeed = (max(data, key= lambda info: info['seed']))['seed']
    itemLen = int(data[0]['dataItemLength'])

    trainingShare = 1 - validationShare - testShare

    def sets(data):
        return setsImpl(data, labels, classCount, itemLen, param)
    
    samples, configs = sets(
        [info for info in data if info['seed'] < maxSeed * trainingShare])
    validationSamples, validationConfigs = sets(
        [info for info in data if info['seed'] >= maxSeed * trainingShare and info['seed'] < maxSeed * (trainingShare + validationShare)])
    testSamples, testConfigs = sets(
        [info for info in data if info['seed'] >= maxSeed * (trainingShare + validationShare)])

    return [samples, configs, validationSamples, validationConfigs, testSamples, testConfigs, classCount, itemLen]

dataPath = 'D:\\Projekty\\flavors-results\\P100-keys-top145\\keysResults.zip'
dataInfoPath = 'D:\\Projekty\\flavors-results\\P100-keys-top145\\dataInfo'
datasetPath = 'D:\\Projekty\\flavors-results\\P100-keys-top145\\findRandomDataset.pkl'

#rawData = fb.throughputs(pd.read_csv(dataPath4, sep=';'))
#fb.bestConfigs(rawData, dataInfoPath4)

#validationShare = 0.2
#testShare = 0.2

#data = load(dataInfoPath)
#samples, configs, validationSamples, validationConfigs, testSamples, testConfigs, classCount, itemLen = dataset(data, validationShare, testShare, 'bestFindRandomConfig')

[samples, configs, validationSamples, validationConfigs, testSamples, testConfigs, classCount, itemLen] = pickle.load(open(datasetPath, 'rb'))

print('Samples count: {0}'.format(len(samples)))
print('Validation samples count: {0}'.format(len(validationSamples)))
print('Test samples count: {0}'.format(len(testSamples)))
print('Class count: {0}'.format(classCount))

#rawData = pd.read_csv(dataPath, sep = ';', compression = 'zip')

#fb.buildThroughput(rawData)
#fb.findThroughput(rawData)
#fb.findRandomThroughput(rawData)

#fb.buildConfigHist(dataInfoPath)
#fb.findRandomConfigHist(dataInfoPath)
#fb.findConfigHist(dataInfoPath)

#plt.show()

#resnetBuilder = resnet.ResnetBuilder()
#model = resnetBuilder.build_resnet_18([1, itemLen, 8], classCount)

#batch_size = 64
#epochs = 1000

#sgd = keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
#model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['acc'])

#history = model.fit(samples, 
#                    configs,
#                    batch_size=batch_size,
#                    epochs=epochs,
#                    verbose=1,
#                    validation_data=(validationSamples, validationConfigs))
 
#score = model.evaluate(validationSamples, validationConfigs, verbose=0)
#print('Test loss on validation set:', score)
#model.save('D:\\Projekty\\flavors-results\\resnet18_1000epochs_sgd.h5')

