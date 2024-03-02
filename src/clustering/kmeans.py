# following
    # https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py

from sklearn.cluster import KMeans

def my_kmeans(X, n_clusters, random_state, n_init=10):
        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=random_state)
        cluster_labels = clusterer.fit_predict(X)
        centers = clusterer.cluster_centers_
        objective_function = clusterer.inertia_
        
        return clusterer, objective_function, cluster_labels, centers
