""" NASBench 101 Benchmark """
import copy
import json
import hashlib
import itertools
import os
from collections import OrderedDict
from pathlib import Path
import numpy as np
from numpy import ndarray
from typing import Callable, Sequence, cast, List, AnyStr, Any, Tuple, Union, Set

from evoxbench.modules import SearchSpace, Evaluator, Benchmark
from nasbench101.models import NASBench101Result  # has to be imported after the init method

__all__ = ['NASBench101SearchSpace', 'NASBench101Evaluator', 'NASBench101Benchmark']

HASH = {'conv3x3-bn-relu': 0, 'conv1x1-bn-relu': 1, 'maxpool3x3': 2}


def get_path(name):
    print(name)
    return str(Path(os.environ.get("EVOXBENCH_MODEL", os.getcwd())) / "nb101" / name)


class Graph:
    def __init__(self) -> None:
        pass

    @staticmethod
    def generate_edges(bits: int) -> Callable:
        return np.vectorize(
            lambda x, y: 0 if x >= y else (bits >> (x + (y * (y - 1) // 2))) % 2 == 1
        )

    @staticmethod
    def is_full(matrix: ndarray) -> bool:
        rows = np.any(np.all(matrix[: np.shape(matrix)[0] - 1, :] == 0, axis=1))
        cols = np.any(np.all(matrix[:, 1:] == 0, axis=0))
        return (not rows) and (not cols)

    @staticmethod
    def edges(matrix: ndarray) -> int:
        return cast(int, np.sum(matrix))

    @staticmethod
    def hash(matrix: ndarray, labels: List[int]) -> AnyStr:
        vertices: int = np.shape(matrix)[0]
        in_edges: List[int] = cast(list, np.sum(matrix, axis=0).tolist())
        out_edges: List[int] = cast(list, np.sum(matrix, axis=1).tolist())
        assert len(in_edges) == len(out_edges) == len(labels)
        hashes: List[AnyStr] = list(map(
            lambda x: hashlib.md5(str(x).encode('utf-8')).hexdigest(),
            list(zip(out_edges, in_edges, labels))
        ))
        for _ in range(vertices):
            _hashes: List[str] = []
            for v in range(vertices):
                in_neighbors = [hashes[w] for w in range(vertices) if matrix[w, v]]
                out_neighbors = [hashes[w] for w in range(vertices) if matrix[v, w]]
                _hashes.append(
                    hashlib.md5(
                        '|'.join((
                            ''.join(sorted(in_neighbors)),
                            ''.join(sorted(out_neighbors)),
                            hashes[v]
                        )).encode('utf-8')
                    ).hexdigest()
                )
            hashes = _hashes
        return hashlib.md5(str(sorted(hashes)).encode('utf-8')).hexdigest()

    @staticmethod
    def permute(graph: ndarray, label: List[Any], permutation: Sequence[int]) -> Tuple[ndarray, List[Any]]:
        forward_perm = zip(permutation, list(range(len(permutation))))
        inverse_perm = [x[1] for x in sorted(forward_perm)]
        new_matrix = np.fromfunction(np.vectorize(lambda x, y: graph[inverse_perm[x], inverse_perm[y]] == 1),
                                     (len(label), len(label)),
                                     dtype=np.int8)
        return cast(ndarray, new_matrix), [label[inverse_perm[i]] for i in range(len(label))]

    @staticmethod
    def is_isomorphic(graph_1: Any, graph_2: Any) -> bool:
        matrix_1, label_1 = np.array(graph_1[0]), graph_1[1]
        matrix_2, label_2 = np.array(graph_2[0]), graph_2[1]
        assert np.shape(matrix_1) == np.shape(matrix_2) and len(label_1) == len(label_2)
        vertices = np.shape(matrix_1)[0]
        for perm in itertools.permutations(range(vertices)):
            _matrix_1, _label_1 = Graph.permute(matrix_1, label_1, perm)
            if np.array_equal(_matrix_1, matrix_2) and _label_1 == label_2:
                return True
        return False

    @staticmethod
    def is_upper_triangular(matrix: ndarray) -> bool:
        for i in range(np.shape(matrix)[0]):
            for j in range(0, i + 1):
                if matrix[i, j] != 0:
                    return False
        return True


class NASBench101Graph(Graph):

    def __init__(self, matrix: Union[ndarray, Sequence[Any]], ops: List[str], data_format: str = 'channels_last'):
        self.__matrix = copy.deepcopy(matrix if isinstance(matrix, ndarray) else np.array(matrix))
        self.__shape = np.shape(self.__matrix)
        assert len(self.__shape) == 2 and self.__shape[0] == self.__shape[1], "Matrix must be square"
        assert self.__shape[0] == len(ops), "Length pf ops must match matrix dimensions"
        assert Graph.is_upper_triangular(self.__matrix), "Matrix must be upper triangular"
        self.__ops = copy.deepcopy(ops)
        self.__data_format = data_format
        self.matrix: ndarray = copy.deepcopy(self.__matrix)
        self.ops: List[Union[str, int]] = copy.deepcopy(self.__ops)
        self.valid: bool = True
        super(NASBench101Graph, self).__init__()
        self.prune()

    def prune(self) -> None:
        vertices_num: int = np.shape(self.__matrix)[0]
        visited: Set[int] = {0}
        frontier: List[int] = [0]
        while frontier:
            top: int = frontier.pop()
            for v in range(top + 1, vertices_num):
                if self.__matrix[top, v] and v not in visited:
                    visited.add(v)
                    frontier.append(v)

        _visited: Set[int] = {vertices_num - 1}
        frontier.clear()
        frontier.append(vertices_num - 1)
        while frontier:
            top: int = frontier.pop()
            for v in range(top):
                if self.__matrix[v, top] and v not in _visited:
                    _visited.add(v)
                    frontier.append(v)
        extraneous: set[int] = set(range(vertices_num)).difference(visited.intersection(_visited))
        if len(extraneous) > vertices_num - 2:
            self.matrix = None
            self.ops = None
            self.valid = False
            return
        self.matrix = np.delete(
            np.delete(self.matrix, list(extraneous), axis=0),
            list(extraneous),
            axis=1
        )
        for index in sorted(extraneous, reverse=True):
            del self.ops[index]

    def hash_spec(self, canonical_ops) -> str:
        labeling = [-1] + [canonical_ops.index(op) for op in self.ops[1:-1]] + [-2]
        return self.hash(self.matrix, labeling)

    def is_valid(self, module_vertices=7, max_edges=9):
        if not self.valid:
            return False

        num_vertices = len(self.ops)
        num_edges = np.sum(self.matrix)

        if num_vertices > module_vertices:
            return False

        if num_edges > max_edges:
            return False

        if self.ops[0] != 'input':
            return False
        if self.ops[-1] != 'output':
            return False
        for op in self.ops[1:-1]:
            if op not in HASH:
                return False
        return True


class NASBench101SearchSpace(SearchSpace):
    """
        NASBench101 API need to be first installed following the official instructions from
        https://github.com/google-research/nasbench
    """

    def __init__(self, **kwargs
                 ):
        super().__init__(**kwargs)
        self.num_vertices = 7
        self.max_edges = 9
        self.edge_spots = int(self.num_vertices * (self.num_vertices - 1) / 2)  # Upper triangular matrix
        self.edge_spots_idx = np.triu_indices(self.num_vertices, 1)
        self.op_spots = int(self.num_vertices - 2)  # Input/output vertices are fixed
        self.allowed_ops = ['conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3']
        self.allowed_edges = [0, 1]  # Binary adjacency matrix

        # upper and lower bound on the decision variables
        self.n_var = int(self.edge_spots + self.op_spots)
        self.lb = [0] * self.n_var
        self.ub = [1] * self.n_var
        self.ub[-self.op_spots:] = [2] * self.op_spots

        # create the categories for each variable
        self.categories = [list(range(a, b + 1)) for a, b in zip(self.lb, self.ub)]

    @property
    def name(self):
        return 'NASBench101SearchSpace'

    def _sample(self, phenotype=True):
        matrix = np.random.choice(self.allowed_edges, size=(self.num_vertices, self.num_vertices))
        matrix = np.triu(matrix, 1)
        ops = np.random.choice(self.allowed_ops, size=self.num_vertices).tolist()
        ops[0] = 'input'
        ops[-1] = 'output'

        if phenotype:
            return {'matrix': matrix, 'ops': ops}
        else:
            return self._encode({'matrix': matrix, 'ops': ops})

    def _encode(self, arch: dict):
        # encode architecture phenotype to genotype
        # a sample arch = {'matrix': matrix, 'ops': ops}, where
        #     # Adjacency matrix of the module
        #     matrix=[[0, 1, 1, 1, 0, 1, 0],  # input layer
        #             [0, 0, 0, 0, 0, 0, 1],  # op1
        #             [0, 0, 0, 0, 0, 0, 1],  # op2
        #             [0, 0, 0, 0, 1, 0, 0],  # op3
        #             [0, 0, 0, 0, 0, 0, 1],  # op4
        #             [0, 0, 0, 0, 0, 0, 1],  # op5
        #             [0, 0, 0, 0, 0, 0, 0]], # output layer
        #     # Operations at the vertices of the module, matches order of matrix
        #     ops=[INPUT, CONV1X1, CONV3X3, CONV3X3, CONV3X3, MAXPOOL3X3, OUTPUT])
        x_edge = np.array(arch['matrix'])[self.edge_spots_idx]
        x_ops = np.empty(self.num_vertices - 2)
        for i, op in enumerate(arch['ops'][1:-1]):
            x_ops[i] = (np.array(self.allowed_ops) == op).nonzero()[0][0]

        return np.concatenate((x_edge, x_ops)).astype(int)

    def _decode(self, x):
        x_edge = x[:self.edge_spots]
        x_ops = x[-self.op_spots:]
        matrix = np.zeros((self.num_vertices, self.num_vertices), dtype=int)
        matrix[self.edge_spots_idx] = x_edge
        ops = ['input'] + [self.allowed_ops[i] for i in x_ops] + ['output']
        return {'matrix': matrix, 'ops': ops}

    def visualize(self, arch):
        raise NotImplementedError


class NASBench101Evaluator(Evaluator):
    def __init__(self,
                 fidelity=108,  # options = [4, 12, 36, 108], i.e. acc measured at different epochs
                 objs='err&params&flops',  # objectives to be minimized
                 ):
        super().__init__(objs)
        self.fidelity = fidelity

    @property
    def name(self):
        return 'NASBench101Evaluator'

    def evaluate(self, archs, objs=None,
                 true_eval=False  # query the true (mean over three runs) performance
                 ):

        if objs is None:
            objs = self.objs

        batch_stats = []
        for i, arch in enumerate(archs):

            stats = OrderedDict()
            model_spec = NASBench101Graph(matrix=arch['matrix'], ops=arch['ops'])

            if model_spec.is_valid():
                arch['matrix'] = np.array(arch['matrix'])
                fingerprint = model_spec.hash_spec(['conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3'])
                ans = NASBench101Result.objects.get(index=fingerprint)

                if true_eval:
                    top1 = np.mean(ans.final_test_accuracy[f"epoch{self.fidelity}"])
                    # top1 = np.mean([v['final_test_accuracy'] for v in ans.result[str(self.fidelity)]])
                else:
                    top1 = np.random.choice(ans.final_validation_accuracy[f"epoch{self.fidelity}"])
                    # top1 = np.random.choice([v['final_validation_accuracy'] for v in ans.result[str(self.fidelity)]])

                params = ans.params
                flops = ans.flops
            else:
                # in case this is not a valid architecture, we assign artificially bad values
                # to make sure if will never survive
                top1 = 0.
                params = np.inf
                flops = np.inf

            if 'err' in objs:
                stats['err'] = 1 - top1

            if 'params' in objs:
                stats['params'] = params

            if 'flops' in objs:
                stats['flops'] = flops

            # # if 'latency' in objs:
            # #     stats['latency'] = fixed['latency']  # in ms
            #
            batch_stats.append(stats)

        return batch_stats


class NASBench101Benchmark(Benchmark):
    def __init__(self,
                 fidelity=108,  # options = [4, 12, 36, 108], i.e. acc measured at different epochs
                 objs='err&params&flops',  # objectives to be minimized
                 pf_file_path=get_path("nb101_pf.json"),  # path to NASBench101 Pareto front json file
                 ps_file_path=get_path("nb101_pf.json"),  # path to NASBench101 Pareto set json file
                 normalized_objectives=True,  # whether to normalize the objectives
                 ):
        search_space = NASBench101SearchSpace()
        evaluator = NASBench101Evaluator(fidelity, objs=objs)
        super().__init__(search_space, evaluator, normalized_objectives)
        self.pf = np.array(json.load(open(pf_file_path, 'r'))[objs])
        self.ps = np.array(json.load(open(ps_file_path, 'r'))[objs])

    @property
    def name(self):
        return 'NASBench101Benchmark'

    @property
    def pareto_front(self):
        return self.pf

    @property
    def pareto_set(self):
        return self.ps

    def debug(self):
        archs = self.search_space.sample(10)
        X = self.search_space.encode(archs)
        F = self.evaluate(X, true_eval=True)
        igd = self.calc_perf_indicator(X, 'igd')
        hv = self.calc_perf_indicator(X, 'hv')
        norm_hv = self.calc_perf_indicator(X, 'normalized_hv')

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
    nb101_pf = "/Users/luzhicha/Dropbox/2022/EvoXBench/python_codes/evoxbench/benchmarks/data/nb101/nb101_pf.json"
    nb101_ps = "/Users/luzhicha/Dropbox/2022/EvoXBench/python_codes/evoxbench/benchmarks/data/nb101/nb101_ps.json"

    benchmark = NASBench101Benchmark(objs='err&flops', pf_file_path=nb101_pf, ps_file_path=nb101_ps,
                                     normalized_objectives=True)
    # print(benchmark.search_space.n_var)
    # exit()
    # pf = benchmark.normalize(benchmark.pareto_front)
    # pf = benchmark.pareto_front
    # print(pf)
    # _pf = benchmark.evaluate(benchmark.search_space.encode(
    #     [benchmark.search_space.repair(arch) for arch in benchmark.pareto_set]), true_eval=True)
    # print(_pf)
    #
    # print(np.sum(np.abs(pf - _pf)))
    # exit()

    benchmark.debug()
