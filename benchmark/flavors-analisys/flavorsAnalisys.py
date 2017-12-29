import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import json

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
pd.set_option('display.expand_frame_repr', False)

def throughputs(data):
    columns = list(data)

    if 'Sort' in columns:
        data['BuildThroughput'] = data['Count'] / (data['Sort'] + data['Reshape'] + data['Build']) * 10 ** 9
    else:
        data['BuildThroughput'] = data['Count'] / data['Build'] * 10 ** 9

    data['FindThroughput'] = data['Count'] / data['Find'] * 10 ** 9
    data['FindRandomThroughput'] = data['RandomCount'] / data['FindRandom'] * 10 ** 9

    if 'FindRandomSorted' in columns:
        data['FindSortedThroughput'] = data['RandomCount'] / data['FindRandomSorted'] * 10 ** 9
    
    return data

def latencies(data):
    columns = list(data)

    if 'Sort' in columns:
        data['BuildLatency'] = (data['Sort'] + data['Reshape'] + data['Build'])
    else:
        data['BuildLatency'] = data['Build']

    data['FindLatency'] = data['Find']
    data['FindRandomLatency'] = data['FindRandom']

    if 'FindRandomSorted' in columns:
        data['FindSortedLatency'] = data['FindRandomSorted']
    
    return data

def aggConfig(rawData, opName, finalAgg, valueName, *indexNames):
    data = pd.pivot_table(
        rawData,
        index=[*indexNames, 'Config'],
        values=['{0}{1}'.format(opName, valueName)],
        aggfunc=[np.mean, np.max, np.min])

    data = pd.pivot_table(
        pd.DataFrame(data.to_records()),
        index=[*indexNames],
        values=[
            "('mean', '{0}{1}')".format(opName, valueName),
            "('amin', '{0}{1}')".format(opName, valueName),
            "('amax', '{0}{1}')".format(opName, valueName)],
        aggfunc=[finalAgg])

    return data

def aggConfigPlot(rawData, opName, finalAgg, valueName, *indexNames):
    data = aggConfig(rawData, opName, finalAgg, valueName, *indexNames)
    plt.figure()
    plt.plot(data.index, data.values)
    plt.legend(['Max', 'Min', 'Mean'])
    plt.grid()

    return

def throughputPlot(rawData, columnName):
    aggConfigPlot(rawData, columnName, np.max, 'Throughput', 'Count')

def latencyPlot(rawData, columnName):
    aggConfigPlot(rawData, columnName, np.min, 'Latency', 'Count')

def buildThroughput(rawData):
    throughputPlot(rawData, 'Build')
    plt.xlabel('Dictionary size [keys]')
    plt.ylabel('Build throughput [keys / sec]')
    plt.title('Build throughput')

def findThroughput(rawData):
    throughputPlot(rawData, 'Find')
    plt.xlabel('Sample size [keys]')
    plt.ylabel('Find throughput [keys / sec]')
    plt.title('Find throughput')

def buildLatency(rawData):
    latencyPlot(rawData, 'Build')
    plt.xlabel('Dictionary size [keys]')
    plt.ylabel('Build latency [ns]')
    plt.title('Build latency')

def findLatency(rawData):
    latencyPlot(rawData, 'Find')
    plt.xlabel('Sample size [keys]')
    plt.ylabel('Find latency [ns]')
    plt.title('Find latency')

def findRandomThroughput(rawData):
    data = pd.DataFrame(aggConfig(
        rawData,
        'FindRandom',
        np.max,
        'Throughput',
        'Count', 'RandomCount').to_records())

    p = plt.figure().gca(projection='3d')
    p.scatter(data['Count'], data['RandomCount'], data['(\'amax\', "(\'amax\', \'FindRandomThroughput\')")'], color='C0')
    #p.scatter(data['Count'], data['RandomCount'], data['(\'amax\', "(\'amin\', \'FindRandomThroughput\')")'], color='C1')
    #p.scatter(data['Count'], data['RandomCount'], data['(\'amax\', "(\'mean\', \'FindRandomThroughput\')")'], color='C2')

    p.set_xlabel('Dictionary size [keys]')
    p.set_ylabel('Batch size [keys]')
    p.set_zlabel('Find random throughput [keys / sec]')
    plt.title('Random find throughput')

def findRandomLatency(rawData):
    data = pd.DataFrame(aggConfig(
        rawData,
        'FindRandom',
        np.min,
        'Latency',
        'Count', 'RandomCount').to_records())

    p = plt.figure().gca(projection='3d')
    #p.scatter(data['Count'], data['RandomCount'], data['(\'amin\', "(\'amax\', \'FindRandomLatency\')")'], color='C0')
    p.scatter(data['Count'], data['RandomCount'], data['(\'amin\', "(\'amin\', \'FindRandomLatency\')")'], color='C1')
    #p.scatter(data['Count'], data['RandomCount'], data['(\'amin\', "(\'mean\', \'FindRandomLatency\')")'], color='C2')

    p.set_xlabel('Dictionary size [keys]')
    p.set_ylabel('Batch size [keys]')
    p.set_zlabel('Find random latency [ns]')
    plt.title('Random find latency')

def buildLatencyOverLength(rawData):
    aggConfigPlot(rawData, 'Build', np.min, 'Latency', 'DataItemLenght')
    plt.xlabel('Key length [bits]')
    plt.ylabel('Build latency [ns]')
    plt.title('Build latency over length')

def buildThroughputOverLength(rawData):
    aggConfigPlot(rawData, 'Build', np.max, 'Throughput', 'DataItemLenght')
    plt.xlabel('Key length [bits]')
    plt.ylabel('Build throughput [keys / sec]')
    plt.title('Build throughput over length')

def findRandomLatencyOverLength(rawData):
    data = pd.DataFrame(aggConfig(
        rawData,
        'FindRandom',
        np.min,
        'Latency',
        'DataItemLenght', 'RandomCount').to_records())

    p = plt.figure().gca(projection='3d')
    p.scatter(data['DataItemLenght'], data['RandomCount'], data['(\'amin\', "(\'amin\', \'FindRandomLatency\')")'], color='C0')

    p.set_xlabel('Data itme length [bits]')
    p.set_ylabel('Batch size [keys]')
    p.set_zlabel('Find random latency [ns]')
    plt.title('Random find latency')

def findRandomThroughputOverLength(rawData):
    data = pd.DataFrame(aggConfig(
        rawData,
        'FindRandom',
        np.max,
        'Throughput',
        'DataItemLenght', 'RandomCount').to_records())

    p = plt.figure().gca(projection='3d')
    p.scatter(data['DataItemLenght'], data['RandomCount'], data['(\'amax\', "(\'amax\', \'FindRandomThroughput\')")'], color='C0')

    p.set_xlabel('Data itme length [bits]')
    p.set_ylabel('Batch size [keys]')
    p.set_zlabel('Find random throughput [keys / sec]')
    plt.title('Random find throughput')

def plot(rawData, indexName, valueName):
    data = pd.pivot_table(
        rawData,
        index=[indexName],
        values=[valueName],
        aggfunc=[np.mean, np.max, np.min])

    plt.figure()
    plt.plot(data.index, data.values)
    plt.grid()
    plt.legend(['Mean', 'Max', 'Min'])

def buildThroughputHost(rawData):
    plot(rawData, 'Count', 'BuildThroughput')
    plt.xlabel('Dictionary size [keys]')
    plt.ylabel('Build throughput [keys / sec]')
    plt.title('CPU build throughput')

def buildLatencyHost(rawData):
    plot(rawData, 'Count', 'BuildLatency')
    plt.xlabel('Dictionary size [keys]')
    plt.ylabel('Build latency [ns]')
    plt.title('CPU build latency')

def findThroughputHost(rawData):
    plot(rawData, 'Count', 'FindThroughput')
    plt.xlabel('Dictionary size [keys]')
    plt.ylabel('Find throughput [sec]')
    plt.title('CPU find throughput')

def findLatencyHost(rawData):
    plot(rawData, 'Count', 'FindLatency')
    plt.xlabel('Dictionary size [keys]')
    plt.ylabel('Find latency [ns]')
    plt.title('CPU find latency')

def findRandomThroughputHost(rawData):
    data = pd.pivot_table(
        rawData,
        index=['Count', 'RandomCount'],
        values=['FindRandomThroughput'],
        aggfunc=[np.max])
    data = pd.DataFrame(data.to_records())

    p = plt.figure().gca(projection='3d')
    p.scatter(data['Count'], data['RandomCount'], data["('amax', 'FindRandomThroughput')"], color='C0')

    p.set_xlabel('Dictionary size [keys]')
    p.set_ylabel('Batch size [keys]')
    p.set_zlabel('Find random throughput [keys / sec]')
    plt.title('Random find throughput')

def findRandomLatencyHost(rawData):
    data = pd.pivot_table(
        rawData,
        index=['Count', 'RandomCount'],
        values=['FindRandomLatency'],
        aggfunc=[np.min])
    data = pd.DataFrame(data.to_records())

    p = plt.figure().gca(projection='3d')
    p.scatter(data['Count'], data['RandomCount'], data["('amin', 'FindRandomLatency')"], color='C0')

    p.set_xlabel('Dictionary size [keys]')
    p.set_ylabel('Batch size [keys]')
    p.set_zlabel('Find random latency [ns]')
    plt.title('Random find latency')

def bestConfigs(rawData, dataInfoPath):

    dataInfo = dict()
    for dataInfoFilePath in os.listdir(dataInfoPath):

        currentFilePath = dataInfoPath + '\\' + dataInfoFilePath
        di = json.load(open(currentFilePath))
        di['path'] = currentFilePath

        di['bestBuildThroughput'] = 0.0
        di['bestFindThroughput'] = 0.0
        di['bestFindRandomThroughput'] = 0.0

        dataInfo[di['count'], di['seed']] = di

    def replaceValue(opName):
        if dataInfo[row['Count'], row['Seed']]['best{0}Throughput'.format(opName)] < row[opName + 'Throughput']:
            dataInfo[row['Count'], row['Seed']]['best{0}Config'.format(opName)] = row['Config']
            dataInfo[row['Count'], row['Seed']]['best{0}Throughput'.format(opName)] = row[opName + 'Throughput']

        return

    for index, row in rawData.iterrows():
        replaceValue('Build')
        replaceValue('Find')
        replaceValue('FindRandom')

    for key, di in dataInfo.items():
        with open(di['path'], 'w') as outfile:
            json.dump(di, outfile)

    return dataInfo

def configsHist(rawData, dataInfoPath, valueName):
    hist = dict()
    for dataInfoFilePath in os.listdir(dataInfoPath):

            currentFilePath = dataInfoPath + '\\' + dataInfoFilePath
            di = json.load(open(currentFilePath))

            if di[valueName] in hist:
                hist[di[valueName]] = hist[di[valueName]] + 1
            else:
                hist[di[valueName]] = 1

    plt.figure()
    plt.bar(range(len(hist)), list(hist.values()), align='center')
    plt.xticks(range(len(hist)), list(hist.keys()), rotation='15', ha='right')

    return hist

def buildConfigHist(rawData, dataInfoPath):
    hist = configsHist(rawData, dataInfoPath, 'bestBuildConfig')
    plt.title('Best build config histogram')

    return hist

def findConfigHist(rawData, dataInfoPath):
    hist = configsHist(rawData, dataInfoPath, 'bestFindConfig')
    plt.title('Best find config histogram')

    return hist

def findRandomConfigHist(rawData, dataInfoPath):
    hist = configsHist(rawData, dataInfoPath, 'bestFindRandomConfig')
    plt.title('Best find random config histogram')

    return hist

#dataPath = 'D:\\Projekty\\flavors-results\\P100-keys\\keysResults.csv'
#dataInfoPath = 'D:\\Projekty\\flavors-results\\P100-keys\\dataInfo'

#dataPath = 'D:\\Projekty\\flavors-results\\P100-keysLen\\keysLenResults.csv'

#dataPath = 'D:\\Projekty\\flavors-results\\GTX1080-keys\\keysResults.csv'
#dataInfoPath = 'D:\\Projekty\\flavors-results\\GTX1080-keys\\dataInfo'

#dataPath = 'D:\\Projekty\\flavors-results\\4790K-hostKeys\\hostResults.csv'

#rawData = throughputs(latencies(pd.read_csv(dataPath, sep=';')))

#buildThroughput(rawData)
#findThroughput(rawData)
#findRandomThroughput(rawData)
#buildLatency(rawData)
#findLatency(rawData)
#findRandomLatency(rawData)

#buildThroughputOverLength(rawData)
#buildLatencyOverLength(rawData)

#findRandomLatencyOverLength(rawData)
#findRandomThroughputOverLength(rawData)

#buildThroughputHost(rawData)
#buildLatencyHost(rawData)
#findThroughputHost(rawData)
#findLatencyHost(rawData)

#findRandomThroughputHost(rawData)
#findRandomLatencyHost(rawData)

#bestConfigs(rawData, dataInfoPath)

#buildConfigHist(rawData, dataInfoPath)
#findConfigHist(rawData, dataInfoPath)
#findRandomConfigHist(rawData, dataInfoPath)

plt.show()

