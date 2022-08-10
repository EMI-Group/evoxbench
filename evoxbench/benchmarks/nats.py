import os
import time
import json
from pathlib import Path

import numpy as np
from typing import List
from numpy import ndarray
from collections import OrderedDict

from evoxbench.modules import SearchSpace, Evaluator, Benchmark
from natsbenchsss.models import NATSBenchResult  # has to be imported after the init method

__all__ = ['NATSBenchSearchSpace', 'NATSBenchEvaluator', 'NATSBenchmark']


def get_path(name):
    return str(Path(os.environ.get("EVOXBENCH_MODEL", os.getcwd())) / "nats" / name)


class NATSBenchSearchSpace(SearchSpace):
    """
        NATS-Bench size search space
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # the allowed channels in each node/layer
        self.allowed_channels: List[int] = [8, 16, 24, 32, 40, 48, 56, 64]

        # upper and lower bound on the decision variables
        self.n_var = 5
        self.lb = [0] * self.n_var
        self.ub = [7] * self.n_var

        # create the categories for each variable
        self.categories = [list(range(a, b + 1)) for a, b in zip(self.lb, self.ub)]

    @property
    def name(self):
        return 'NATSBenchSearchSpace'

    def _sample(self, phenotype=True):
        tmp = np.random.choice(self.allowed_channels, self.n_var)
        arch = ':'.join(map(str, tmp.tolist()))
        return arch if phenotype else self._encode(arch)

    def _encode(self, arch: str):
        # encode architecture phenotype to genotype
        # a sample architecture
        # '64:56:8:24:16:24:48:8'
        x_chns = np.empty(self.n_var, dtype=np.int64)
        for i, chns in enumerate(arch.split(':')):
            x_chns[i] = (np.array(self.allowed_channels) == int(chns)).nonzero()[0][0]
        return x_chns

    def _decode(self, x: ndarray) -> str:
        return ':'.join(map(str, [self.allowed_channels[i] for i in x]))

    def visualize(self, arch):
        raise NotImplementedError

    # ------------------------ Following functions are specific to NASBench301 ------------------------- #


class NATSBenchEvaluator(Evaluator):
    def __init__(self,
                 fidelity=90,  # options = [1, 12, 90], i.e. acc measured at different training epochs
                 objs='err&params&flops',  # objectives to be minimized
                 dataset='cifar10',  # ['cifar10', 'cifar100', 'ImageNet16-120']
                 ):
        super().__init__(objs)
        self.engine = NATSBenchResult
        self.fidelity = str(fidelity)
        self.dataset = dataset

    @property
    def name(self):
        return 'NATSBenchEvaluator'

    def evaluate(self, archs, objs=None,
                 # this dataset currently do not support noisy evaluation
                 true_eval=False  # query the true (mean over multiple runs) performance
                 ):

        if objs is None:
            objs = self.objs

        batch_stats = []
        infos = {result.phenotype: result for result in NATSBenchResult.objects.filter(phenotype__in=archs)}
        for arch in archs:
            info = infos[arch]
            stats = OrderedDict()
            if self.dataset == 'cifar10':
                if true_eval:
                    computed = info.cifar10
                else:
                    computed = info.cifar10_valid

            elif self.dataset == 'cifar100':
                computed = info.cifar100

            elif self.dataset == 'ImageNet16-120':
                computed = info.ImageNet16_120

            else:
                raise KeyError

            if true_eval:
                top1 = np.mean([v['test-accuracy'] for v in computed['hp{}'.format(self.fidelity)]])
            else:
                top1 = np.random.choice(computed['hp{}'.format(self.fidelity)])['valid-accuracy']

            if 'err' in objs:
                stats['err'] = 100 - top1

            if 'params' in objs:
                stats['params'] = info.cost[self.dataset]['hp{}'.format(self.fidelity)]['params']

            if 'flops' in objs:
                stats['flops'] = info.cost[self.dataset]['hp{}'.format(self.fidelity)]['flops']

            if 'latency' in objs:
                stats['latency'] = info.cost[self.dataset]['hp{}'.format(self.fidelity)]['latency']  # in ms

            batch_stats.append(stats)

        return batch_stats


class NATSBenchmark(Benchmark):
    def __init__(self,
                 fidelity=90,  # options = [1, 12, 90, 200], i.e. acc measured at different training epochs
                 objs='err&params&flops',  # objectives to be minimized
                 dataset='cifar10',  # ['cifar10', 'cifar100', 'ImageNet16-120']
                 pf_file_path=get_path("nats_pf.json"),  # path to the Pareto front json file
                 ps_file_path=get_path("nats_ps.json"),  # path to the Pareto set json file
                 normalized_objectives=True,  # whether to normalize the objectives
                 ):
        search_space = NATSBenchSearchSpace()
        evaluator = NATSBenchEvaluator(fidelity, objs, dataset)
        super().__init__(search_space, evaluator, normalized_objectives)

        self.pf = np.array(json.load(open(pf_file_path, 'r'))[objs])
        self.ps = np.array(json.load(open(ps_file_path, 'r'))[objs])

    @property
    def name(self):
        return 'NASBench201Benchmark'

    @property
    def pareto_front(self):
        return self.pf

    @property
    def pareto_set(self):
        return self.ps

    def debug(self):
        archs = self.search_space.sample(10)
        X = self.search_space.encode(archs)
        F = self.evaluate(X, true_eval=False)
        print(F)

        start = time.time()
        igd = self.calc_perf_indicator(X, 'igd')
        print("igd time: {}".format(time.time() - start))

        start = time.time()
        hv = self.calc_perf_indicator(X, 'hv')
        print("hv time: {}".format(time.time() - start))

        start = time.time()
        norm_hv = self.calc_perf_indicator(X, 'normalized_hv')
        print("normalized hv time: {}".format(time.time() - start))

        # ps_X = self.search_space.encode(self.ps)
        ps_igd = self.calc_perf_indicator(self.pareto_set, 'igd')
        ps_hv = self.calc_perf_indicator(self.pareto_set, 'hv')
        ps_norm_hv = self.calc_perf_indicator(self.pareto_set, 'normalized_hv')

        print(archs)
        print(X)
        print(F)
        print(igd)
        print(hv)
        print(norm_hv)

        print("PF IGD: {}, this number should be really close to 0".format(ps_igd))
        print(ps_hv)
        print("PF normalized HV: {}, this number should be really close to 1".format(ps_norm_hv))


if __name__ == '__main__':
    nats_pf = "/Users/luzhicha/Dropbox/2022/EvoXBench/python_codes/evoxbench/benchmarks/data/nats/nats_pf.json"
    nats_ps = "/Users/luzhicha/Dropbox/2022/EvoXBench/python_codes/evoxbench/benchmarks/data/nats/nats_ps.json"

    benchmark = NATSBenchmark(
        90, objs='err&params&flops', dataset='cifar10', pf_file_path=nats_pf, ps_file_path=nats_ps,
        normalized_objectives=True
    )

    benchmark.debug()
