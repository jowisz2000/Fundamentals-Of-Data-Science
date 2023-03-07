from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import rand_score, accuracy_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score

# I needed to change commas to spaces so that np.loadtxt could work
with open('wine.data', 'r') as file:
    filedata = file.read()

filedata = filedata.replace(',', '  ')

with open('wine.data', 'w') as file:
    file.write(filedata)

wine_data = np.loadtxt('wine.data')

# outlier detection
wine_data = wine_data[(np.abs(stats.zscore(wine_data)) < 3).all(axis=1)]

labels = wine_data[:, 0]
wine_data = wine_data[:, 1:]

# reducing dimensions of data
pca = PCA(n_components=2)
pca.fit(wine_data)
transformed_data = pca.transform(wine_data)

# data visualization
plt.scatter(transformed_data[:, 0], transformed_data[:, 1], c=labels)
plt.xlabel('1st dimension')
plt.ylabel('2nd dimension')
plt.title('Reduced dimension data')
plt.savefig('reduced_dimension_data.png')
plt.close()

# clustering data with unreduced dimensions
kmeans = KMeans(n_clusters=3, n_init=10)
kmeans.fit(wine_data)
print('Clustering')
print(f'Accuracy of clustering unreduced dimensions data: {rand_score(labels, kmeans.labels_)}')
print(f'Accuracy of clustering unreduced dimensions data'
      f' (Silhouette score):{silhouette_score(wine_data,kmeans.labels_)}')

# clustering data with reduced dimensions
kmeans = KMeans(n_clusters=3, n_init=10)
kmeans.fit(transformed_data)
print(f'Accuracy of clustering reduced data: {rand_score(labels, kmeans.labels_)}')
print(f'Accuracy of clustering reduced data(Silhouette score): {silhouette_score(transformed_data, kmeans.labels_)}')

# clasification
print('\nClassification')
n, metric = 5, 'euclidean'
accuracies = []
kF = KFold(n_splits=5)

for i, (train_index, test_index) in enumerate(kF.split(wine_data)):
    neighbours = KNeighborsClassifier(n_neighbors=n, metric=metric, random_state=40)
    neighbours.fit(wine_data[train_index], labels[train_index])
    accuracies.append(accuracy_score(neighbours.predict([wine_data[test_index]][0]), labels[test_index]))
print(f'average accuracy score with {n} neighbours for unreduced dimensions data:{mean(accuracies)}')

accuracies = []
kF = KFold(n_splits=5)

for i, (train_index, test_index) in enumerate(kF.split(transformed_data)):
    neighbours = KNeighborsClassifier(n_neighbors=n, metric=metric)
    neighbours.fit(transformed_data[train_index], labels[train_index])
    accuracies.append(accuracy_score(neighbours.predict([transformed_data[test_index]][0]), labels[test_index]))
print(f'average accuracy score with {n} neighbours for reduced dimensions data:{mean(accuracies)}')

clf = LogisticRegression(max_iter=10000)
X_train, X_test, y_train, y_test = train_test_split(wine_data, labels, test_size=0.2, random_state=44)
# Fit the classifier on the training data
clf.fit(X_train, y_train)
# Make predictions on the testing data
y_pred = clf.predict(X_test)
print(f'accuracy score in method suggested by chat-gpt (unchanged data): {accuracy_score(y_test, y_pred)}')

clf = LogisticRegression(max_iter=10000)
X_train, X_test, y_train, y_test = train_test_split(transformed_data, labels, test_size=0.2, random_state=44)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(f'accuracy score in method suggested by chat-gpt (two dimensional data): {accuracy_score(y_test, y_pred)}')
