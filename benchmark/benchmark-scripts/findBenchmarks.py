import pandas as pd
import matplotlib.pyplot as plt
import subprocess as sb
import json as js
import gzip
import shutil
import time
import os

def runKeysFind(caption, benchPath, configPath, resultsPath):
    sb.check_call([benchPath, configPath]);
    config = js.load(open(configPath))
    outName = time.strftime('%Y-%m-%d_%H-%M-%S') + '_' + caption + '.csv.gz'

    with open(config['resultFilePath'], 'rb') as f_in, gzip.open(resultsPath + outName, 'wb') as f_out:
       shutil.copyfileobj(f_in, f_out)

    data = pd.read_csv(config['resultFilePath'], sep=';')
    os.remove(config['resultFilePath'])
    
    return data

def calculateThroughputs(data):
    data['BuildThroughput'] = data['Count'] / (data['Sort'] + data['Reshape'] + data['Build']) * 10**9
    data['FindThroughput'] = data['Count'] / data['Find'] * 10**9
    data['FindRandomThroughput'] = data['Count'] / data['FindRandom'] * 10**9
    data['FindSortedThroughput'] = data['Count'] / data['FindRandomSorted'] * 10**9
    
    return data
    
def drawChart(maxData, meanData, minData, column, caption, groupColumn, chartKind):
    data = pd.DataFrame()
    data[groupColumn] = maxData[groupColumn]
    data[caption + '_max'] = maxData[column]
    data[caption + '_mean'] = meanData[column]
    data[caption + '_min'] = minData[column]
    data.plot(x=groupColumn, title=column, grid='on', kind=chartKind);
    
    if chartKind != 'bar':
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
def drawCharts(data, caption, groupColumn, chartKind = 'line'):
    groupedDataMax = data.groupby(groupColumn).max().reset_index(groupColumn)
    groupedDataMin = data.groupby(groupColumn).min().reset_index(groupColumn)
    groupedDataMean = data.groupby(groupColumn).mean().reset_index(groupColumn)

    drawChart(groupedDataMax, groupedDataMean, groupedDataMin, 'BuildThroughput', caption, groupColumn, chartKind)
    drawChart(groupedDataMax, groupedDataMean, groupedDataMin, 'FindThroughput', caption, groupColumn, chartKind)
    drawChart(groupedDataMax, groupedDataMean, groupedDataMin, 'FindRandomThroughput', caption, groupColumn, chartKind)
    drawChart(groupedDataMax, groupedDataMean, groupedDataMin, 'FindSortedThroughput', caption, groupColumn, chartKind)

def findBenchmark(caption, benchPath, configPath, resultsPath):
    data = runKeysFind(caption, benchPath, configPath, resultsPath)
    data = calculateThroughputs(data)
    
    drawCharts(data, caption, 'Count')
    drawCharts(data, caption, 'Seed')
    drawCharts(data, caption, 'Config', 'bar')
    
def keysFindBenchmark(caption = 'keysFindRun', benchPath = './../benchmark-run/Release/flavors-benchmarks-run', configPath = '../configurations/keysFind.json', resultsPath = '../../results/keysFind/'):
    findBenchmark(caption, benchPath, configPath, resultsPath)
  
def masksFindBenchmark(caption = 'masksFindRun', benchPath = './../benchmark-run/Release/flavors-benchmarks-run', configPath = '../configurations/masksFind.json', resultsPath = '../../results/masksFind/'):
    findBenchmark(caption, benchPath, configPath, resultsPath)
    
def benchmarkHistory(resultsPath, measures, params): 
    
    dataFrames = {p: {m: pd.DataFrame() for m in measures} for p in params}
    
    for fileName in os.listdir(resultsPath):
        if fileName.endswith('.gz'):
            fileData = calculateThroughputs(pd.read_csv(resultsPath + fileName, sep=';', compression='gzip'))
            
            for p in params:
                groupedData = fileData.groupby(p).mean()
            
                for m in measures:
                    (dataFrames[p][m])[os.path.splitext(os.path.splitext(fileName)[0])[0]] = groupedData[m]
    
    for p in params:  
        for m in measures:
            if p != 'Config':
                (dataFrames[p][m]).plot(title = m + ' by ' + p, grid = 'on')
                plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
            else:
                (dataFrames[p][m]).plot(title = m + ' by ' + p, grid = 'on', kind = 'bar')
            
def keysBenchmarkHistory(resultsPath = '../../results/keysFind/', measures = ['BuildThroughput', 'FindThroughput', 'FindRandomThroughput'], params = ['Count', 'Config', 'Seed']):
    benchmarkHistory(resultsPath, measures, params)
    
def masksBenchmarkHistory(resultsPath = '../../results/masksFind/', measures = ['BuildThroughput', 'FindThroughput', 'FindRandomThroughput'], params = ['Count', 'Config', 'Seed']):
    benchmarkHistory(resultsPath, measures, params)
    