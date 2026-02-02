from abc import ABC, abstractmethod
import numpy as np


class ClusterAlgorithm(ABC):
    @abstractmethod
    def predict(self, data: np.ndarray) -> np.ndarray:
        ...