import numpy as np
from sklearn.cluster import KMeans
from .cluster_algorithm import ClusterAlgorithm


class KMeansClusterAlgorithm(ClusterAlgorithm):
    def __init__(self, n_clusters: int = 5, random_state: int = 42):
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)

    def predict(self, data: np.ndarray) -> np.ndarray:
        return self.kmeans.fit_predict(data)