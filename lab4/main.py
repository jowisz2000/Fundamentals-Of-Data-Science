import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics.cluster import rand_score
from statistics import mean

data = np.loadtxt('seeds_dataset.txt')
lastColumn = data[:, -1]
data = data[:, :-1]

for n in range(3, 10):
    for metric in ['euclidean', 'cosine', 'manhattan']:
        neighbours = KNeighborsClassifier(n_neighbors=n, metric=metric)
        neighbours.fit(data, lastColumn)
        accuracies = []
        kF = KFold(n_splits=5)

        for i, (train_index, test_index) in enumerate(kF.split(data)):
            accuracies.append(rand_score(neighbours.predict([data[test_index]][0]), lastColumn[test_index]))

        print(f'metric={metric}, average accuracy with {n} neighbours:{mean(accuracies)}')
    print('-----------------------------------------------------')
