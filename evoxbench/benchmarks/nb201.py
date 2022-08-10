import json
import os
from pathlib import Path

import numpy as np
from typing import List
from numpy import ndarray
from collections import OrderedDict

from evoxbench.modules import SearchSpace, Evaluator, Benchmark
from nasbench201.models import NASBench201Result  # has to be imported after the init method

__all__ = ['NASBench201SearchSpace', 'NASBench201Evaluator', 'NASBench201Benchmark']


def get_path(name):
    return str(Path(os.environ.get("EVOXBENCH_MODEL", os.getcwd())) / "nb201" / name)


class NASBench201SearchSpace(SearchSpace):
    """
        NASBench201 topology search space
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_vertices: int = 4
        self.edge_spots: int = self.num_vertices * (self.num_vertices - 1) // 2

        # the operation must be none rather than zeroize
        self.allowed_ops: List[str] = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']

        # upper and lower bound on the decision variables
        self.n_var = self.edge_spots
        self.lb = [0] * self.n_var
        self.ub = [4] * self.n_var

        # create the categories for each variable
        self.categories = [list(range(a, b + 1)) for a, b in zip(self.lb, self.ub)]

    @property
    def name(self):
        return 'NASBench201SearchSpace'

    def _sample(self):
        x = np.random.choice(len(self.allowed_ops), self.edge_spots)
        return self._decode(x)

    def _encode(self, arch: str):
        # encode architecture phenotype to genotype
        # a sample architecture
        # '|none~0|+|avg_pool_3x3~0|nor_conv_3x3~1|+|nor_conv_1x1~0|nor_conv_3x3~1|nor_conv_1x1~2|'
        ops = []
        for node in arch.split('+'):
            op = [o.split('~')[0] for o in node.split('|') if o]
            ops.extend(op)
        x_ops = np.empty(self.edge_spots, dtype=np.int64)
        for i, op in enumerate(ops):
            x_ops[i] = (np.array(self.allowed_ops) == op).nonzero()[0][0]
        return x_ops

    def _decode(self, x: ndarray) -> str:
        _x = x.tolist()
        result, index = [], 0
        for i in range(1, self.num_vertices):
            tmp = []
            for j in range(i):
                tmp.append(f"{self.allowed_ops[_x[index]]}~{j}")
                index += 1
            result.append(f"|{'|'.join(tmp)}|")
        return '+'.join(result)

    def visualize(self, arch):
        raise NotImplementedError


class NASBench201Evaluator(Evaluator):
    def __init__(self,
                 fidelity=200,  # options = [1, 12, 90, 200], i.e. acc measured at different training epochs
                 # objs='err&params&flops',  # objectives to be minimized
                 objs='err&params&flops&edgegpu_latency&edgegpu_energy&'
                      'eyeriss_latency&eyeriss_energy&eyeriss_arithmetic_intensity',  # objectives to be minimized
                 dataset='cifar10',  # ['cifar10', 'cifar100', 'ImageNet16-120']
                 # hardware='fpga',  # ['edgegpu', 'raspi4', 'edgetpu', 'pixel3', 'eyeriss', 'fpga']
                 ):
        super().__init__(objs)
        self.engine = NASBench201Result
        # self.hardware = hardware
        self.fidelity = str(fidelity)
        self.dataset = dataset

    @property
    def name(self):
        return 'NASBench201Evaluator'

    def evaluate(self, archs, objs=None,
                 true_eval=False  # query the true (mean over multiple runs) performance
                 ):

        if objs is None:
            objs = self.objs
        objs = set(objs.split('&'))
        batch_stats = []
        infos = {result.phenotype: result for result in NASBench201Result.objects.filter(phenotype__in=archs)}
        for arch in archs:
            info = infos[arch]
            stats = OrderedDict()
            fixed = info.cost12[self.dataset] if self.fidelity == '12' else info.cost200[self.dataset]
            if true_eval:
                top1 = np.mean([v['test-accuracy'] for v in
                                info.more_info[self.dataset]['hp{}'.format(self.fidelity)]])  # mean test accuracy
            else:
                computed = [v['valid-accuracy'] for v in info.more_info[
                    'cifar10-valid' if self.dataset == 'cifar10' else self.dataset][
                    'hp{}'.format(self.fidelity)]]
                top1 = np.random.choice(computed)  # randomly chosen from trials

            if 'err' in objs:
                stats['err'] = 100 - top1
            if 'params' in objs:
                stats['params'] = fixed['params']  # in M
            if 'flops' in objs:
                stats['flops'] = fixed['flops']  # in M

            # the remaining obj in objs should be latency, energy or arithmetic_intensity
            for obj in objs:
                if 'latency' in obj:
                    hardware = obj.split('_')[0]
                    assert hardware in ['edgegpu', 'eyeriss', 'fpga'], "the requested hardware do not support latency"
                    stats[obj] = info.hw_info[self.dataset][obj]
                if 'energy' in obj:
                    hardware = obj.split('_')[0]
                    assert hardware in ['edgegpu', 'eyeriss', 'fpga'], "the requested hardware do not support energy"
                    stats[obj] = info.hw_info[self.dataset][obj]
                if 'arithmetic_intensity' in obj:
                    hardware = obj.split('_')[0]
                    assert hardware in ['eyeriss'], "only Eyeriss support arithmetic intensity"
                    stats[obj] = info.hw_info[self.dataset][obj]
                # ms for latency, mJ for energy, OPs/Byte for arithmetic_intensity

            batch_stats.append(stats)

        return batch_stats


class NASBench201Benchmark(Benchmark):
    def __init__(self,
                 fidelity=200,  # options = [12, 200], i.e. acc measured at different training epochs
                 objs='err&params&flops',  # objectives to be minimized
                 dataset='cifar10',  # ['cifar10', 'cifar100', 'ImageNet16-120']
                 # hardware='fpga',  # ['edgegpu', 'raspi4', 'edgetpu', 'pixel3', 'eyeriss', 'fpga']
                 pf_file_path=get_path("nb201_pf.json"),  # path to the Pareto front json file
                 ps_file_path=get_path("nb201_pf.json"),  # path to the Pareto set json file
                 normalized_objectives=True,  # whether to normalize the objectives
                 ):
        search_space = NASBench201SearchSpace()
        evaluator = NASBench201Evaluator(fidelity, objs, dataset)
        super().__init__(search_space, evaluator, normalized_objectives)

        # # process objectives w.r.t hardware
        # if 'latency' in objs:
        #     objs = objs.replace('latency', '{}_latency'.format(hardware))
        #
        # if 'energy' in objs:
        #     assert hardware in ['edgegpu', 'eyeriss', 'fpga'], \
        #         "the requested hardware do not support energy"
        #     objs = objs.replace('energy', '{}_energy'.format(hardware))
        #
        # if 'arithmetic_intensity' in objs:
        #     assert hardware in ['eyeriss'], \
        #         "only Eyeriss support arithmetic intensity"
        #     objs = objs.replace('arithmetic_intensity', '{}_arithmetic_intensity'.format(hardware))

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

        igd = self.calc_perf_indicator(X, 'igd')
        hv = self.calc_perf_indicator(X, 'hv')
        norm_hv = self.calc_perf_indicator(X, 'normalized_hv')

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


# ------------------------ Following functions are specific to NASBench201 ------------------------- #


if __name__ == '__main__':
    nb201_pf = "/Users/luzhicha/Dropbox/2022/EvoXBench/python_codes/evoxbench/benchmarks/data/nb201/nb201_pf.json"
    nb201_ps = "/Users/luzhicha/Dropbox/2022/EvoXBench/python_codes/evoxbench/benchmarks/data/nb201/nb201_ps.json"

    benchmark = NASBench201Benchmark(
        200, objs='err&params&flops&edgegpu_latency&edgegpu_energy&'
                  'eyeriss_latency&eyeriss_energy&eyeriss_arithmetic_intensity', dataset='cifar10',
        # hardware='eyeriss',
        pf_file_path=nb201_pf, ps_file_path=nb201_ps, normalized_objectives=True)

    # print(benchmark.search_space.n_var)
    # exit()

    # pf = benchmark.normalize(benchmark.pareto_front)
    # _pf = benchmark.evaluate(benchmark.search_space.encode(benchmark.pareto_set), true_eval=True)
    #
    # print(np.sum(np.abs(pf - _pf)))
    # exit()
    benchmark.debug()
