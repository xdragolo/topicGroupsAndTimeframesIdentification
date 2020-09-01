from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
#https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html

# vstup:
#     max, min: hranicni pocet n_clustru pro ktere se ma vypocitat mean silhouette
#     X: document - frequancy matrix
#     n_clusters: pocet shluku

def silhouette_plot_tendency(min, max,X, spectral = False,name = './figures/silhouette_tendency.png'):
    x = []
    y = []
    if spectral:
        for n_clusters in range(min,max):
            x.append(n_clusters)
            #Kmeans
            clusterer = SpectralClustering(n_clusters=n_clusters,affinity='nearest_neighbors')
            cluster_labels = clusterer.fit_predict(X)
            # report average silhouette
            silhouette_avg = silhouette_score(X, cluster_labels)
            y.append(silhouette_avg)
            print("For n_clusters =", n_clusters,
                  "The average silhouette_score is :", silhouette_avg)
            title = "Spectral Clustering"
    else:
        for n_clusters in range(min,max):
            x.append(n_clusters)
            #Kmeans
            clusterer = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=150, n_init=1, random_state=10)
            cluster_labels = clusterer.fit_predict(X)
            # report average silhouette
            silhouette_avg = silhouette_score(X, cluster_labels)
            y.append(silhouette_avg)
            print("For n_clusters =", n_clusters,
                  "The average silhouette_score is :", silhouette_avg)
            title = "K-means Clustering"

    plt.plot(x,y)
    # print(x)
    # print(y)
    plt.xlabel("n_clusters")
    plt.title('Tendency of avarage silhouette score for '+ title)
    plt.ylabel("average silhouette score")
    if name:
        plt.savefig(name)
    plt.show()