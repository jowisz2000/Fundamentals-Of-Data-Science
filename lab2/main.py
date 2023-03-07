import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data = pd.read_csv('seeds_dataset.txt', sep='\t', on_bad_lines='skip')

labels = data.iloc[:,-1]
data = data.iloc[:,:-1]

pca = PCA(n_components=2)
reduced_data = pca.fit_transform(data)

plt.scatter(reduced_data[:,0], reduced_data[:,1], c=labels)
plt.savefig('pca.png')

tsne = TSNE()
reduced_data = tsne.fit_transform(data)
plt.scatter(reduced_data[:,0], reduced_data[:,1], c=labels)
plt.savefig('tsne.png')