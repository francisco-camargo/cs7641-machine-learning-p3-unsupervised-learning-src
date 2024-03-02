# Run GMM fitting and labeling
from sklearn.mixture import GaussianMixture
def em_gmm(X, n_clusters, random_state):
    max_iter=100
    n_init=1
    clusterer = GaussianMixture(
        n_components=n_clusters,
        max_iter=max_iter,
        n_init=n_init,
        random_state=random_state)
    clusterer.fit(X)
    cluster_labels = clusterer.predict(X)
    # cluster_probabilities = clusterer.predict_proba(X) 
        # If we want to use clusters as features, we could use the probabilities instead of just the labels
    centers = clusterer.means_
    return clusterer, cluster_labels, centers


# Plotting
    # https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_covariances.html#sphx-glr-auto-examples-mixture-plot-gmm-covariances-py
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.colors as mcolors
import numpy as np

def mask_data(X, masking_data):
    mask = np.ma.masked_where(np.column_stack((masking_data,masking_data)), X)
    masked_array = np.ma.compressed(mask)
    return masked_array
    

alpha = 0.2
def plot_gmm(X, cluster_labels, gmm, n_clusters, x_dim_0, x_dim_1):
    # default_colors = [value for key, value in mcolors.TABLEAU_COLORS.items()] 
    fig, ax = plt.subplots()
    
    for cluster_idx in range(n_clusters):
        local_X = X[cluster_labels==cluster_idx]
        # color = colors[cluster_idx]
        p = plt.plot(local_X[:, x_dim_0], local_X[:, x_dim_1], '.', label=f'{cluster_idx}')
        color = p[0].get_color() # get str value of color
        color = colors.colorConverter.to_rgb(color) # convert to tuple value of color
        make_ellipses(gmm, cluster_idx, color, ax)


def make_ellipses(gmm, cluster_idx, color, ax):
    if gmm.covariance_type == "full":
        covariances = gmm.covariances_[cluster_idx][:2, :2]
    elif gmm.covariance_type == "tied":
        covariances = gmm.covariances_[:2, :2]
    elif gmm.covariance_type == "diag":
        covariances = np.diag(gmm.covariances_[cluster_idx][:2])
    elif gmm.covariance_type == "spherical":
        covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[cluster_idx]
    v, w = np.linalg.eigh(covariances)
    u = w[0] / np.linalg.norm(w[0])
    angle = np.arctan2(u[1], u[0])
    angle = 180 * angle / np.pi  # convert to degrees
    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
    ell = mpl.patches.Ellipse(gmm.means_[cluster_idx, :2], v[0], v[1], angle=180 + angle, color=color, alpha=alpha)
    ell.set_clip_box(ax.bbox)
    # ell.set_alpha(0.5)
    ax.add_artist(ell)
    ax.set_aspect("equal", "datalim")