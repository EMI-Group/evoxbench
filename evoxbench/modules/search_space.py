import numpy as np
from abc import ABC, abstractmethod

__all__ = ['SearchSpace']


class SearchSpace(ABC):
    def __init__(self, **kwargs):
        # attributes below have to be filled by child class
        self.n_var = None
        self.lb = None
        self.ub = None
        self.categories = None

    @property
    @abstractmethod
    def name(self):
        raise NotImplementedError

    @abstractmethod
    def _sample(self):
        """ method to randomly create an architecture phenotype """
        raise NotImplementedError

    def sample(self, n_samples):
        subnets = []
        for _ in range(n_samples):
            subnets.append(self._sample())
        return subnets

    @abstractmethod
    def _encode(self, arch):
        """ method to convert architectural string to search decision variable vector """
        raise NotImplementedError

    def encode(self, archs):
        X = []
        for arch in archs:
            X.append(self._encode(arch))
        return np.array(X)

    @abstractmethod
    def _decode(self, x):
        """ method to convert decision variable vector to architectural string """
        raise NotImplementedError

    def decode(self, X):
        archs = []
        for x in X:
            archs.append(self._decode(x))
        return archs

    @abstractmethod
    def visualize(self, arch):
        """ method to visualize an architecture """
        raise NotImplementedError
