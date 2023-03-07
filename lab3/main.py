import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import rand_score
from sklearn.decomposition import PCA

data = np.loadtxt('seeds_dataset.txt')
toCompare = data[:, -1]
data = data[:, 0:-1]


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


linkages = ['ward', 'complete', 'average', 'single']
metrics = ['euclidean', 'manhattan', 'cosine']
for metric in metrics:
    clustering = AgglomerativeClustering(metric=metric, linkage='single', distance_threshold=0, n_clusters=None)
    clustering=clustering.fit(data)
    # print(clustering.labels_)
    plot_dendrogram(clustering, truncate_mode="level", p=3)
    plt.title(metric)
    plt.savefig(metric+'.png')
    plt.close()


kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto").fit(data)
print('Rand score for data with 7 dimensions:' , rand_score(kmeans.labels_, toCompare))

pca = PCA(n_components=2)
reduced_data = pca.fit_transform(data)
kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto").fit(reduced_data)
print('Rand score for data with 2 dimensions:' , rand_score(kmeans.labels_, toCompare))

# kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto")
#