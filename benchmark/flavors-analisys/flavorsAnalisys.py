import benchmarks as fb
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
import os
import json 
import numpy as np

#using only first device
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def load(path):
    data = []
    for fileName in os.listdir(path):
        with open(os.path.join(path, fileName)) as file:
            data.append(json.load(file))

    return data

def getConfig(data):
    return ''.join(map(str, data['bestBuildConfig']))

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

def dataset(data):
    stats = []
    configs = []

    validationStats = []
    validationConfigs = []

    confgisLabels = dict()
    itemLen = int(data[0]['dataItemLength'])

    for info in data:
        config = getConfig(info)

        if not config in confgisLabels:
            confgisLabels[config] = len(confgisLabels)

        if int(info['seed']) < 45:
            configs.append(confgisLabels[config])
            stats.extend(getStats(info))
        else:
            validationConfigs.append(confgisLabels[config])
            validationStats.extend(getStats(info))

    samples = np.array(stats).reshape(len(configs), itemLen * 8)
    configs = np.array(configs)

    validationSamples = np.array(validationStats).reshape(len(validationConfigs), itemLen * 8)
    validationConfigs = np.array(validationConfigs)

    classCount = max(max(configs), max(validationConfigs)) + 1
    configs = keras.utils.to_categorical(configs, classCount)
    validationConfigs = keras.utils.to_categorical(validationConfigs, classCount)

    return samples, configs, validationSamples, validationConfigs, classCount, itemLen

dataInfoPath = 'D:\\Projekty\\flavors-results\\P100-keys-4\\dataInfo'

data = load(dataInfoPath)
samples, configs, validationSamples, validationConfigs, classCount, itemLen = dataset(data)

print('\nData loaded')
print('Samples count: {0}'.format(len(samples)))
print('Validation samples count: {0}'.format(len(validationSamples)))
print('Class count: {0}'.format(classCount))

