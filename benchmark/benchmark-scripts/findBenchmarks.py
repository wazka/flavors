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
    
def drawChart(maxData, meanData, minData, column, caption):
    data = pd.DataFrame()
    data['Count'] = maxData['Count']
    data[caption + '_max'] = maxData[column]
    data[caption + '_mean'] = meanData[column]
    data[caption + '_min'] = minData[column]
    data.plot(x='Count', title=column, grid='on');
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

def findBenchmark(caption, benchPath, configPath, resultsPath):
    data = runKeysFind(caption, benchPath, configPath, resultsPath)
    data = calculateThroughputs(data)

    byCountDataMax = data.groupby('Count').max().reset_index('Count')
    byCountDataMin = data.groupby('Count').min().reset_index('Count')
    byCountDataMean = data.groupby('Count').mean().reset_index('Count')

    drawChart(byCountDataMax, byCountDataMean, byCountDataMin, 'BuildThroughput', caption)
    drawChart(byCountDataMax, byCountDataMean, byCountDataMin, 'FindThroughput', caption)
    drawChart(byCountDataMax, byCountDataMean, byCountDataMin, 'FindRandomThroughput', caption)
    drawChart(byCountDataMax, byCountDataMean, byCountDataMin, 'FindSortedThroughput', caption)
    
def keysFindBenchmark(caption = 'keysFindRun', benchPath = './../benchmark-run/Release/flavors-benchmarks-run', configPath = '../configurations/keysFind.json', resultsPath = '../../results/keysFind/'):
    findBenchmark(caption, benchPath, configPath, resultsPath)
  
def masksFindBenchmark(caption = 'masksFindRun', benchPath = './../benchmark-run/Release/flavors-benchmarks-run', configPath = '../configurations/masksFind.json', resultsPath = '../../results/masksFind/'):
    findBenchmark(caption, benchPath, configPath, resultsPath)  