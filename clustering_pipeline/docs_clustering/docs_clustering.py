from typing import Dict
import numpy as np
from .cluster_algorithm.cluster_algorithm import ClusterAlgorithm

from .cluster_algorithm.kmeans import KMeansClusterAlgorithm


def cluster_documents(
    documents: Dict[str, Dict],
    clustering_algorithm: ClusterAlgorithm = KMeansClusterAlgorithm(),
) -> Dict[str, Dict]:
    """
    Cluster documents based on their feature vectors.
    """
    
    doc_names = list(documents.keys())

    X = np.array([
        documents[name]["vector"]
        for name in doc_names
    ], dtype=float)

    labels = clustering_algorithm.predict(X)

    clusters: Dict[str, Dict] = {}
    
    for name, label in zip(doc_names, labels):
        clusters[name] = {}
        clusters[name]["cluster_id"] = int(label)

    return clusters