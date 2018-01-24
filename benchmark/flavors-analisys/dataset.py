import os
import json
import numpy as np
import keras.utils as ku

def load(paths):
    if type(paths) is str:
        return _loadFromFile(paths)
    else:
        data = []
        for path in paths:
            data.extend(_loadFromFile(path))
        return data

def _loadFromFile(path):
    data = []
    paths = list(os.listdir(path))

    for i, fileName in enumerate(paths):
        with open(os.path.join(path, fileName)) as file:
            data.append(json.load(file))
        if i % 1000 == 0:
            print('Files loaded: {0}'.format(i))
        

    return data

def _appendOptional(stats, pos, param):
    if pos[param] == None:
        stats.append(0.0)
    else:
        stats.append(pos[param])

def _getStats(data):
    stats = []
    for pos in data['dataInfo']:
        stats.append(float(pos['mean']))
        stats.append(float(pos['max']))
        stats.append(float(pos['min']))
        stats.append(float(pos['std']))
        stats.append(float(pos['n']))
        _appendOptional(stats, pos, 'kurtosis')
        _appendOptional(stats, pos, 'skewness')
        _appendOptional(stats, pos, 'variance')
    return stats

def _getConfig(info, param, itemLen):
    config = list(map(int, info[param].replace('{', '').replace('}', '').split(',')))

    return config

def _setsImpl(data, itemLen, param):
    stats = []
    rawConfigs = []

    for info in data:
        stats.extend(_getStats(info))
        rawConfigs.append(_getConfig(info,param, itemLen))

    maxConfigLen = len(max(rawConfigs, key = lambda c: len(c)))

    configs = []
    for c in rawConfigs:
        c.extend([0] * (maxConfigLen - len(c)))
        configs.extend(c)

    samples = ku.normalize(np.array(stats).reshape(len(data), itemLen * 8), axis = 0)
    configs = ku.normalize(np.array(configs).reshape(len(data), maxConfigLen), axis = 0)

    return samples, configs, maxConfigLen

def build(data, validationShare, testShare, param):
    maxSeed = (max(data, key= lambda info: info['seed']))['seed']
    itemLen = int(data[0]['dataItemLength'])
    trainingShare = 1 - validationShare - testShare

    def sets(data):
        return _setsImpl(data, itemLen, param)
    
    samples, configs, maxLen = sets(
        [info for info in data if info['seed'] < maxSeed * trainingShare])
    validationSamples, validationConfigs, maxValidationLen = sets(
        [info for info in data if info['seed'] >= maxSeed * trainingShare and info['seed'] < maxSeed * (trainingShare + validationShare)])
    testSamples, testConfigs, maxTrainingLen = sets(
        [info for info in data if info['seed'] >= maxSeed * (trainingShare + validationShare)])

    if maxLen != maxValidationLen or maxLen != maxTrainingLen:
        print('Warning: max config len differs in subsets')

    return [samples, configs, validationSamples, validationConfigs, testSamples, testConfigs, itemLen, maxLen]
