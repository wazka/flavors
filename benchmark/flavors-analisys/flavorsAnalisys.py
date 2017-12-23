import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculateThroughputs(data):
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

dataPath = 'D:\\Projekty\\flavors-results\\P100-keys\\keysResults.csv'
#dataPath = 'D:\\Projekty\\flavors-results\\4790K-hostKeys\\hostResults.csv'

rawData = calculateThroughputs(
                    pd.read_csv(
                        dataPath, 
                        sep=';'))

data = pd.DataFrame(
    pd.pivot_table(
        pd.DataFrame(
            pd.pivot_table(
                rawData, 
                index=['Config', 'Count', 'RandomCount'], 
                values=['BuildThroughput', 'FindThroughput', 'FindRandomThroughput'],
                aggfunc=[np.mean])
                    .to_records())
                        .rename(
                            index=str, 
                            columns={"('mean', 'BuildThroughput')": "BuildThroughput", "('mean', 'FindThroughput')": "FindThroughput",  "('mean', 'FindRandomThroughput')": "FindRandomThroughput"}), 
            index=['Count'], 
            values=['BuildThroughput', 'FindThroughput', 'FindRandomThroughput'], 
            aggfunc=[np.max]).to_records()).rename(index=str, 
                        columns={"('amax', 'BuildThroughput')": "BuildThroughput", "('amax', 'FindThroughput')": "FindThroughput", "('amax', 'FindRandomThroughput')": "FindRandomThroughput"})

print(data)

data.plot(
    x='Count', 
    y='BuildThroughput',
    title='BuildThroughput', 
    grid='on', 
    kind='line', 
    ax=plt.gca())

plt.figure()
data.plot(
    x='Count', 
    y='FindThroughput',
    title='FindThroughput', 
    grid='on', 
    kind='line', 
    ax=plt.gca())

plt.figure()
data.plot(
    x='Count', 
    y='FindRandomThroughput',
    title='FindRandomThroughput', 
    grid='on', 
    kind='line', 
    ax=plt.gca())
plt.show()

