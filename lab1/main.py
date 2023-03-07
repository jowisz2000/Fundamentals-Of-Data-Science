import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from sklearn.neighbors import NearestNeighbors


file = pd.read_csv('haberman.data')

df = pd.DataFrame(file)
fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
firstLabel = [df.iloc[i,:] for i in range(len(df.iloc[:,0])) if df.iloc[i,-1]==2]
dfFirstLabel = pd.DataFrame(firstLabel)
secondLabel = [df.iloc[i,:] for i in range(len(df.iloc[:,0])) if df.iloc[i,-1]==1]
dfSecondLabel = pd.DataFrame(secondLabel)

ax.scatter3D(df.iloc[:,0],df.iloc[:,1],df.iloc[:,2])

ax.scatter3D(dfFirstLabel.iloc[:,0],dfFirstLabel.iloc[:,1],dfFirstLabel.iloc[:,2], facecolor='gold')
ax.scatter3D(dfSecondLabel.iloc[:,0],dfSecondLabel.iloc[:,1],dfSecondLabel.iloc[:,2], facecolor='limegreen')
plt.xlabel('age')
plt.ylabel('operation year')
ax.set_zlabel('nodes detected')
plt.title("simple 3D scatter plot")
plt.savefig('task1.png')
plt.close()

df.drop(df.columns[[-1]], axis=1, inplace=True)

task2 = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
task2.to_csv('task2.csv')


task3 = NearestNeighbors(n_neighbors=5)

