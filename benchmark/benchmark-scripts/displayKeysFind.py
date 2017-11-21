import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import subprocess as sb
import json as js
import gzip
import shutil
import time
import os

benchPath = './../benchmark-run/Release/flavors-benchmarks-run'
configPath = '../configurations/keysFind.json'

def runBenchmark(caption):
    sb.check_call([benchPath, configPath]);
    config = js.load(open(configPath))
    outPath = time.strftime('%Y-%m-%d_%H-%M-%S') + '_' + caption + '.csv.gz'

    with open(config['resultFilePath'], 'rb') as f_in, gzip.open(outPath, 'wb') as f_out:
       shutil.copyfileobj(f_in, f_out)

    data = pd.read_csv(config['resultFilePath'], sep=';')
    os.remove(config['resultFilePath'])
    
    return data, outPath

def calculateThroughputs(data):
    data['BuildThroughput'] = data['Count'] / (data['Sort'] + data['Reshape'] + data['Build']) * 10**9
    data['FindThroughput'] = data['Count'] / data['Find'] * 10**9
    data['FindRandomThroughput'] = data['Count'] / data['FindRandom'] * 10**9
    data['FindSortedThroughput'] = data['Count'] / data['FindRandomSorted'] * 10**9
    
    return data
    
def drawChart(maxData, meanData, minData, column):
    data = pd.DataFrame()
    data['Count'] = maxData['Count']
    data[caption + '_max'] = maxData[column]
    data[caption + '_mean'] = meanData[column]
    data[caption + '_min'] = minData[column]
    data.plot(x='Count', title=column, grid='on');
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

#running benchmark
caption = 'first_run'
[data, outPath] = runBenchmark(caption)

data = calculateThroughputs(data)

#printing by count charts
byCountDataMax = data.groupby('Count').max().reset_index('Count')
byCountDataMin = data.groupby('Count').min().reset_index('Count')
byCountDataMean = data.groupby('Count').mean().reset_index('Count')

drawChart(byCountDataMax, byCountDataMean, byCountDataMin, 'BuildThroughput')
drawChart(byCountDataMax, byCountDataMean, byCountDataMin, 'FindThroughput')
drawChart(byCountDataMax, byCountDataMean, byCountDataMin, 'FindRandomThroughput')
drawChart(byCountDataMax, byCountDataMean, byCountDataMin, 'FindSortedThroughput')
