import numpy as np
from numpy import ndarray
from abc import ABC, abstractmethod

from .evaluator import Evaluator
from .search_space import SearchSpace

__all__ = ['Benchmark']


class Benchmark(ABC):
    def __init__(self,
                 search_space: SearchSpace,
                 evaluator: Evaluator,
                 normalized_objectives=False,
                 **kwargs):
        self.search_space = search_space
        self.evaluator = evaluator
        self.normalized_objectives = normalized_objectives

    @property
    @abstractmethod
    def name(self):
        raise NotImplementedError

    @property
    def pareto_front(self):
        return None

    @property
    def pareto_set(self):
        return None

    @property
    def _utopian_point(self):
        return None

    @property
    def _nadir_point(self):
        return None

    @property
    def utopian_point(self):
        if self._utopian_point is not None:
            return np.array(self._utopian_point)
        elif self.pareto_front is not None:
            return np.min(self.pareto_front, axis=0)
        else:
            return None

    @property
    def nadir_point(self):
        if self._nadir_point is not None:
            return np.array(self._nadir_point)
        elif self.pareto_front is not None:
            return np.max(self.pareto_front, axis=0)
        else:
            return None

    @property
    def _hv_ref_point(self):
        return None

    @property
    def hv_ref_point(self):
        # reference point for calculating hypervolume
        if self._hv_ref_point is not None:
            return self._hv_ref_point
        elif self.nadir_point is not None:
            return self.nadir_point
        else:
            return None

    def normalize(self, F: np.array):
        """ method to normalize the objectives  """
        assert self.utopian_point is not None, "Missing Pareto front or utopian point for normalization"
        assert self.nadir_point is not None, "Missing Pareto front or nadir point for normalization"
        return (F - self.utopian_point) / (self.nadir_point - self.utopian_point)

    def to_matrix(self, batch_stats: dict) -> ndarray:
        # convert performance dict to objective matrix
        return np.array([list(v.values()) for v in batch_stats])

    def evaluate(self, X, **kwargs):
        # convert genotype X to architecture phenotype
        archs = self.search_space.decode(X)

        # query for performance
        batch_stats = self.evaluator.evaluate(archs, **kwargs)

        # convert performance dict to objective matrix
        F = self.to_matrix(batch_stats)

        # normalize objective matrix
        if self.normalized_objectives:
            F = self.normalize(F)

        return F

    def calc_perf_indicator(self, inputs, indicator='igd'):

        # The performance indicator that calculates IGD and HV are from [pymoo](https://pymoo.org/).
        try:
            from pymoo.indicators.igd import IGD
            from pymoo.indicators.hv import HV
            from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
        except ImportError:
            raise ImportError('please first install pymoo from https://pymoo.org/')

        assert indicator in ['igd', 'hv', 'normalized_hv'], "The requested performance indicator is not supported"

        if indicator == 'igd':
            assert self.pareto_front is not None, "Pareto front needs to be defined before IGD calculation"

        if indicator == 'hv':
            assert self.hv_ref_point is not None, "A reference point need to be defined before HV calculation"

        if indicator == 'igd':
            # normalize Pareto front
            pf = self.normalize(self.pareto_front)
            metric = IGD(pf=pf)
        elif 'hv' in indicator:
            hv_ref_point = self.normalize(self.hv_ref_point)
            metric = HV(ref_point=hv_ref_point)
        else:
            raise KeyError("the requested performance indicator is not define")

        if isinstance(inputs[0], ndarray):
            # re-evaluate the true performance
            F = self.evaluate(inputs, true_eval=True)  # use true/mean accuracy

        else:
            batch_stats = self.evaluator.evaluate(inputs, true_eval=True)
            F = self.to_matrix(batch_stats)

        if not self.normalized_objectives:
            F = self.normalize(F)  # in case the benchmark evaluator does not normalize objs by default

        # filter out the non-dominated solutions
        nd_front = NonDominatedSorting().do(F, only_non_dominated_front=True)
        performance = metric(F[nd_front])

        if indicator == 'normalized_hv' and self.pareto_front is not None:
            hv_norm = metric(self.normalize(self.pareto_front))
            performance = performance / hv_norm

        return performance
