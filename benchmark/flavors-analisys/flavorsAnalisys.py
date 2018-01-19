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

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()

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

    paths = list(os.listdir(path))
    printProgressBar(0, len(paths), prefix = 'Loading data progress:', suffix = 'Complete', length = 50)

    for i, fileName in enumerate(paths):
        with open(os.path.join(path, fileName)) as file:
            data.append(json.load(file))

        printProgressBar(i + 1, len(paths), prefix = 'Loading data progress:', suffix = 'Complete', length = 50)

    return data

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

def getConfig(info, param, itemLen):

    config = list(map(int, info[param].replace('{', '').replace('}', '').split(',')))
    config.extend([0] * (itemLen - len(config)))

    return config

def setsImpl(data, itemLen, param):
    stats = []
    configs = []

    for info in data:
        stats.extend(getStats(info))
        configs.extend(getConfig(info,param, itemLen))

    samples = np.array(stats).reshape(len(data), itemLen, 8, 1)
    configs = np.array(configs).reshape(len(data), itemLen)

    return samples, configs

def dataset(data, validationShare, testShare, param):
    maxSeed = (max(data, key= lambda info: info['seed']))['seed']
    itemLen = int(data[0]['dataItemLength'])
    trainingShare = 1 - validationShare - testShare

    def sets(data):
        return setsImpl(data, itemLen, param)
    
    samples, configs = sets(
        [info for info in data if info['seed'] < maxSeed * trainingShare])
    validationSamples, validationConfigs = sets(
        [info for info in data if info['seed'] >= maxSeed * trainingShare and info['seed'] < maxSeed * (trainingShare + validationShare)])
    testSamples, testConfigs = sets(
        [info for info in data if info['seed'] >= maxSeed * (trainingShare + validationShare)])

    return [samples, configs, validationSamples, validationConfigs, testSamples, testConfigs, itemLen]

dataPath = 'D:\\Projekty\\flavors-results\\P100-keys-top10\\keysResults.zip'
dataInfoPath = 'D:\\Projekty\\flavors-results\\P100-keys-top10\\dataInfo'

findRandomDatasetPath = 'D:\\Projekty\\flavors-results\\P100-keys-top10\\findRandomDataset.pkl'
buildDatasetPath = 'D:\\Projekty\\flavors-results\\P100-keys-top10\\buildDataset.pkl'

#rawData = fb.throughputs(pd.read_csv(dataPath4, sep=';'))
#fb.bestConfigs(rawData, dataInfoPath4)

#validationShare = 0.2
#testShare = 0.2

#data = load(dataInfoPath)
#buildDataset = dataset(data, validationShare, testShare, 'bestBuildConfig')
#findRandomDataset = dataset(data, validationShare, testShare, 'bestFindRandomConfig')

[samples, configs, validationSamples, validationConfigs, testSamples, testConfigs, itemLen] = pickle.load(open(findRandomDatasetPath, 'rb'))

#pickle.dump(buildDataset, open(buildDatasetPath, 'wb'))
#pickle.dump(findRandomDataset, open(findRandomDatasetPath, 'wb'))

#print('Samples count: {0}'.format(len(samples)))
#print('Validation samples count: {0}'.format(len(validationSamples)))
#print('Test samples count: {0}'.format(len(testSamples)))
#print('Class count: {0}'.format(classCount))

#rawData = pd.read_csv(dataPath, sep = ';', compression = 'zip')

#fb.buildThroughput(rawData)
#fb.findThroughput(rawData)
#fb.findRandomThroughput(rawData)

#fb.buildConfigHist(dataInfoPath)
#fb.findRandomConfigHist(dataInfoPath)
#fb.findConfigHist(dataInfoPath)

#plt.show()

resnetBuilder = resnet.ResnetBuilder()
model = resnetBuilder.build_resnet_101([1, itemLen, 8], itemLen)

batch_size = 64
epochs = 500

#sgd = keras.optimizers.SGD(lr=0.3, momentum=0.01, decay=0.01, nesterov=False)
model.compile(loss='mean_squared_error', optimizer='rmsprop')

history = model.fit(samples, 
                    configs,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(validationSamples, validationConfigs))
 
#score = model.evaluate(validationSamples, validationConfigs, verbose=0)
#print('Test loss on validation set:', score)
#model.save('D:\\Projekty\\flavors-results\\P100-keys-top10\\resnet18_100epochs_adam.h5')

#samples = samples.reshape(len(samples), itemLen * 8)
#validationSamples = validationSamples.reshape(len(validationSamples), itemLen * 8)

#model = Sequential()
#model.add(Dense(256, activation='sigmoid', input_shape=[itemLen* 8]))
#model.add(Dense(256, activation='sigmoid'))
#model.add(Dense(512, activation='sigmoid'))
#model.add(Dense(512, activation='sigmoid'))
#model.add(Dense(1024, activation='sigmoid'))
#model.add(Dense(2048, activation='sigmoid'))
#model.add(Dense(4096, activation='sigmoid'))
#model.add(Dense(4096, activation='sigmoid'))
#model.add(Dense(2048, activation='sigmoid'))
#model.add(Dense(2048, activation='sigmoid'))
#model.add(Dense(1024, activation='sigmoid'))
#model.add(Dense(1024, activation='sigmoid'))
#model.add(Dense(512, activation='sigmoid'))
#model.add(Dense(512, activation='sigmoid'))
#model.add(Dense(256, activation='sigmoid'))
#model.add(Dense(256, activation='sigmoid'))
#model.add(Dense(128, activation='sigmoid'))
#model.add(Dense(128, activation='sigmoid'))
#model.add(Dense(64, activation='sigmoid'))
#model.add(Dense(64, activation='sigmoid'))
#model.add(Dense(32, activation='sigmoid'))
#model.add(Dense(32))

#model = Sequential()
#model.add(Dense(itemLen* 8,init='uniform', activation='linear', input_shape=[itemLen* 8]))
#model.add(Dense(itemLen* 8,init='uniform', activation='linear'))
#model.add(Dense(itemLen* 8,init='uniform', activation='linear'))
#model.add(Dense(itemLen,init='uniform', activation='linear'))
 
##model.summary()
 
#model.compile(loss='mean_absolute_error',
#              optimizer='rmsprop')

#history = model.fit(samples, 
#                    configs,
#                    batch_size=batch_size,
#                    epochs=epochs,
#                    verbose=1,
#                    validation_data=(validationSamples, validationConfigs))

