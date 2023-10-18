"""
This type stub file was generated by pyright.
"""

from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin

"""Hierarchical k-means clustering."""
def hierarchical_k_means(X, n_clusters, init=..., batch_size=..., n_init=..., max_no_improvement=..., verbose=..., random_state=...): # -> ndarray[Any, dtype[Any]]:
    """Use a recursive k-means to cluster X.

    First clustering in sqrt(n_clusters) parcels,
    and Kmeans a second time on each parcel.

    Parameters
    ----------
    X: ndarray (n_samples, n_features)
        Data to cluster

    n_clusters: int,
        The number of clusters to find.

    init : {'k-means++', 'random' or an ndarray}
        Method for initialization, defaults to 'k-means++':
        'k-means++' : selects initial cluster centers for k-means
        clustering in a smart way to speed up convergence. See section
        Notes in k_init for more details.
        'random': choose k observations (rows) at random from data for
        the initial centroids.
        If an ndarray is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.

    batch_size : int, optional, default: 1000
        Size of the mini batches. (Kmeans performed through MiniBatchKMeans)

    n_init : int, default=10
        Number of random initializations that are tried.
        In contrast to KMeans, the algorithm is only run once, using the
        best of the ``n_init`` initializations as measured by inertia.

    max_no_improvement : int, default: 10
        Control early stopping based on the consecutive number of mini
        batches that does not yield an improvement on the smoothed inertia.
        To disable convergence detection based on inertia, set
        max_no_improvement to None.

    random_state : int, RandomState instance or None (default)
        Determines random number generation for centroid initialization and
        random reassignment. Use an int to make the randomness deterministic.

    Returns
    -------
    labels : list of ints (len n_features)
        Parcellation of features in clusters
    """
    ...

class HierarchicalKMeans(BaseEstimator, ClusterMixin, TransformerMixin):
    """Hierarchical KMeans.

    First clusterize the samples into big clusters. Then clusterize the samples
    inside these big clusters into smaller ones.

    Parameters
    ----------
    n_clusters: int
        The number of clusters to find.

    init : {'k-means++', 'random' or an ndarray}
        Method for initialization, defaults to 'k-means++':

        * 'k-means++' : selects initial cluster centers for k-means
          clustering in a smart way to speed up convergence. See section
          Notes in k_init for more details.

        * 'random': choose k observations (rows) at random from data for
          the initial centroids.

        * If an ndarray is passed, it should be of shape (n_clusters,
          n_features) and gives the initial centers.

    batch_size : int, optional, default: 1000
        Size of the mini batches. (Kmeans performed through MiniBatchKMeans)

    n_init : int, default=10
        Number of random initializations that are tried.
        In contrast to KMeans, the algorithm is only run once, using the
        best of the ``n_init`` initializations as measured by inertia.

    max_no_improvement : int, default: 10
        Control early stopping based on the consecutive number of mini
        batches that does not yield an improvement on the smoothed inertia.
        To disable convergence detection based on inertia, set
        max_no_improvement to None.

    random_state : int, RandomState instance or None (default)
        Determines random number generation for centroid initialization and
        random reassignment. Use an int to make the randomness deterministic.

    scaling: bool, optional (default False)
        If scaling is True, each cluster is scaled by the square root of its
        size during transform(), preserving the l2-norm of the image.
        inverse_transform() will apply inversed scaling to yield an image with
        same l2-norm as input.

    verbose: int, optional (default 0)
        Verbosity level.

    Attributes
    ----------
    `labels_ `: ndarray, shape = [n_features]
        cluster labels for each feature.

    `sizes_`: ndarray, shape = [n_features]
        It contains the size of each cluster.

    """
    def __init__(self, n_clusters, init=..., batch_size=..., n_init=..., max_no_improvement=..., verbose=..., random_state=..., scaling=...) -> None:
        ...
    
    def fit(self, X, y=...): # -> Self@HierarchicalKMeans:
        """Compute clustering of the data.

        Parameters
        ----------
        X: ndarray, shape = [n_features, n_samples]
            Training data.
        y: Ignored

        Returns
        -------
        self
        """
        ...
    
    def transform(self, X, y=...): # -> NDArray[floating[Any]] | NDArray[float64]:
        """Apply clustering, reduce the dimensionality of the data.

        Parameters
        ----------
        X: ndarray, shape = [n_features, n_samples]
            Data to transform with the fitted clustering.

        Returns
        -------
        X_red: ndarray, shape = [n_clusters, n_samples]
            Data reduced with agglomerated signal for each cluster
        """
        ...
    
    def inverse_transform(self, X_red):
        """Send the reduced 2D data matrix back to the original feature \
        space (voxels).

        Parameters
        ----------
        X_red: ndarray , shape = [n_clusters, n_samples]
            Data reduced with agglomerated signal for each cluster

        Returns
        -------
        X_inv: ndarray, shape = [n_features, n_samples]
            Data reduced expanded to the original feature space
        """
        ...
    

