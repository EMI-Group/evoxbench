from abc import ABC, abstractmethod

__all__ = ['Evaluator']


class Evaluator(ABC):
    def __init__(self,
                 objs='err&params',  # objectives to be minimized
                 **kwargs):
        self.objs = objs

    @property
    @abstractmethod
    def name(self):
        raise NotImplementedError

    @property
    def n_objs(self):
        return len(self.objs.split('&'))

    @abstractmethod
    def evaluate(self, archs, **kwargs):
        """ method to evaluate the population """
        raise NotImplementedError
