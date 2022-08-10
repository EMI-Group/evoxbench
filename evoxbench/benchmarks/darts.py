import json
import warnings
import numpy as np
from collections import namedtuple
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder
from pathlib import Path
import os
from collections import OrderedDict

from evoxbench.modules import SearchSpace, Evaluator, Benchmark, MLPPredictor

warnings.filterwarnings("ignore")

__all__ = ['DARTSSearchSpace', 'DARTSEvaluator', 'DARTSBenchmark', 'Genotype']

PRIMITIVES = [
    'skip_connect',
    'avg_pool_3x3',
    'max_pool_3x3',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5',
]

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
Genotype_cell = namedtuple('CellGenotype', 'cell concat')


def get_path(name):
    return str(Path(os.environ.get("EVOXBENCH_MODEL", os.getcwd())) / "darts" / name)


class Profiler:
    def __init__(self, data_file=get_path("profile_data.json")):
        with open(data_file, "r") as f:
            data = json.load(f)
        self.params = data["params"]
        self.macs = data["macs"]

    def _gen_key(self, layer, C, stride, affine, input_shape=None):
        keys = ["op", layer, "C:" + str(C), "stride:" + str(stride), "affine:" + str(affine)]
        if input_shape:
            if not isinstance(input_shape, tuple):
                input_shape = tuple(input_shape)
            keys.append("input_shape:" + str(input_shape))
        key = "/".join(keys)
        return key

    def get_params_op(self, layer, C, stride, affine):
        key = self._gen_key(layer, C, stride, affine)
        if not key in self.params:
            raise Exception(key + "not found")
        return self.params[key]

    def get_macs_op(self, layer, C, stride, affine, input_shape):
        key = self._gen_key(layer, C, stride, affine, input_shape)
        if not key in self.macs:
            raise Exception(key + "not found")
        return self.macs[key]


class CellProfiler:
    """ supporting func for calculating mparams and flops """

    def __init__(self,
                 profiler,  # profiling methods
                 genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):

        self.profiler = profiler
        self.reduction_prev = reduction_prev
        self.C_prev_prev = C_prev_prev
        self.C_prev = C_prev
        self.C = C

        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        assert len(op_names) == len(indices)

        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = []
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1

            op = (name, C, stride, True)
            self._ops += [op]
        self._indices = indices

    def macs(self, s0, s1):
        macs = 0
        s0 = s0.copy()
        s1 = s1.copy()

        if self.reduction_prev:
            assert s0[1] == self.C_prev_prev
            s0[1] = self.C
            s0[2] = s0[2] // 2
            s0[3] = s0[3] // 2
        else:
            assert s0[1] == self.C_prev_prev
            s0[1] = self.C

        assert s1[1] == self.C_prev
        s1[1] = self.C

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]].copy()
            h2 = states[self._indices[2 * i + 1]].copy()

            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]

            macs = macs + self.profiler.get_macs_op(op1[0], op1[1], op1[2], op1[3], h1)
            macs = macs + self.profiler.get_macs_op(op2[0], op2[1], op2[2], op2[3], h2)

            if op1[2] == 2:  # stride = 2
                h1[2] = h1[2] // 2
                h1[3] = h1[3] // 2
            if op2[2] == 2:
                h2[2] = h2[2] // 2
                h2[3] = h2[3] // 2
            # print(h1,h2)
            assert h1 == h2

            s = h1.copy()
            states += [s]

        s = states[self._concat[0]].copy()
        for i in self._concat[1:]:
            s[1] = s[1] + states[i][1]
        return s, macs

    def params(self):
        params = 0
        for op in self._ops:
            op_param = self.profiler.get_params_op(op[0], op[1], op[2], op[3])
            params = params + op_param
        return params


class ComplexityPredictor:
    """ method for predicting #Params and #FLOPs """

    def __init__(self, data_file_path=get_path("profile_data.json"),
                 C=32, layers=8, auxiliary=False):

        self.profiler = Profiler(data_file_path)

        self.base_macs = 79733440  # offset:some fixed ops and layers
        self.base_params = 392426

        self._layers = layers
        self._auxiliary = auxiliary
        self.drop_path_prob = 0  # no difference when inferring

        stem_multiplier = 3
        C_curr = stem_multiplier * C
        self.C = C
        self.C_curr = C_curr

    def _compile(self, genotype):
        C_prev_prev, C_prev, C_curr = self.C_curr, self.C_curr, self.C
        layers = self._layers

        self.cells = []
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False

            cell = CellProfiler(self.profiler, genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr

    def _macs(self, input):
        macs = 0
        assert input[1] == 3
        input = list(input)
        input[1] = self.C_curr

        s0 = input.copy()
        s1 = input.copy()
        for i, cell in enumerate(self.cells):
            s2, cmacs = cell.macs(s0, s1)
            macs = macs + cmacs
            s0, s1 = s1, s2
        return macs

    def _params(self):
        params = 0
        for cell in self.cells:
            params = params + cell.params()
        return params

    def predict_params(self, genotype):
        self._compile(genotype)
        return self._params() + self.base_params

    def predict_macs(self, genotype, input_shape=(1, 3, 32, 32)):
        self._compile(genotype)
        return self._macs(input_shape) + self.base_macs


class DARTSFeatureEncoder(object):
    def __init__(self,
                 categories,
                 n_blocks=4,  # number of blocks in a cell
                 n_cells=2,  # number of cells to search
                 ):
        self.n_blocks = n_blocks
        self.n_ops = len(PRIMITIVES)
        self.n_cells = n_cells
        self.allowed_ops = PRIMITIVES
        self.categories = categories
        self.oh_enc_len = 0
        for cat in categories:
            self.oh_enc_len += len(cat)

    def arch2x(self, arch):
        # the same as Search Space encode
        x = []
        # normal cell
        for unit in arch.normal:
            x.append((np.array(self.allowed_ops) == unit[0]).nonzero()[0][0])
            x.append(unit[1])

        # reduction cell
        for unit in arch.reduce:
            x.append((np.array(self.allowed_ops) == unit[0]).nonzero()[0][0])
            x.append(unit[1])

        return np.array(x)

    def archs2feature(self, archs):
        X = [self.arch2x(arch) for arch in archs]

        # Transform using one-hot encoding.
        features = np.zeros((len(X), self.oh_enc_len))

        for j, x in enumerate(X):
            base = 0
            for i in range(len(x)):
                try:
                    idx = self.categories[i].index(x[i])
                except:
                    raise ValueError("Found unknown categories [{}] in column {}.".format(x[i], i))
                features[j][base + idx] = 1.0
                base += len(self.categories[i])
        return features


class DARTSSearchSpace(SearchSpace):
    """ DARTS search space """

    def __init__(self,
                 n_blocks=4,  # number of blocks in a cell
                 n_cells=2,  # number of cells to search
                 **kwargs
                 ):
        super().__init__(**kwargs)

        self.n_blocks = n_blocks
        self.n_ops = len(PRIMITIVES)
        self.n_cells = n_cells
        self.allowed_ops = PRIMITIVES

        # upper and lower bound on the decision variables
        self.n_var = int(n_blocks * 2 * 2 * n_cells)  # 2 unit per block, 2 bits per unit, and n_cells
        self.lb = [0] * self.n_var
        self.ub = [1] * self.n_var

        h = 1
        for b in range(0, self.n_var // 2, 4):
            self.ub[b] = self.n_ops - 1
            self.ub[b + 1] = h
            self.ub[b + 2] = self.n_ops - 1
            self.ub[b + 3] = h
            h += 1
        self.ub[self.n_var // 2:] = self.ub[:self.n_var // 2]

        # create the categories for each variable
        self.categories = [list(range(a, b + 1)) for a, b in zip(self.lb, self.ub)]

    @property
    def name(self):
        return 'DARTSSearchSpace'

    def _sample(self, phenotype=True):
        x = []
        for i in range(1, self.n_var, 4):
            inp = np.random.choice(self.categories[i], 2, replace=False)
            x.extend([np.random.choice(len(self.allowed_ops)), inp[0],
                      np.random.choice(len(self.allowed_ops)), inp[1]])
        # x = [np.random.choice(options) for options in self.categories]
        return self._decode(x) if phenotype else np.array(x)

    def _encode(self, arch: Genotype):
        x = []
        # normal cell
        for unit in arch.normal:
            x.append((np.array(self.allowed_ops) == unit[0]).nonzero()[0][0])
            x.append(unit[1])

        # reduction cell
        for unit in arch.reduce:
            x.append((np.array(self.allowed_ops) == unit[0]).nonzero()[0][0])
            x.append(unit[1])

        return np.array(x)

    def _decode(self, x):
        genome = self._convert(x)
        # decodes genome to architecture
        normal_cell = self._decode_cell(genome[0])
        reduce_cell = self._decode_cell(genome[1])

        return Genotype(
            normal=normal_cell.cell, normal_concat=normal_cell.concat,
            reduce=reduce_cell.cell, reduce_concat=reduce_cell.concat)

    def visualize(self, arch):
        raise NotImplementedError

    # ------------------------ Following functions are specific to DARTS serach space ------------------------- #
    def _convert_cell(self, cell_bit_string):
        # convert cell bit-string to genome
        tmp = [cell_bit_string[i:i + 2] for i in range(0, len(cell_bit_string), 2)]
        return [tmp[i:i + 2] for i in range(0, len(tmp), 2)]

    def _convert(self, bit_string):
        # convert network bit-string (norm_cell + redu_cell) to genome
        norm_gene = self._convert_cell(bit_string[:len(bit_string) // 2])
        redu_gene = self._convert_cell(bit_string[len(bit_string) // 2:])
        return [norm_gene, redu_gene]

    def _decode_cell(self, cell_genome):

        cell, cell_concat = [], list(range(2, len(cell_genome) + 2))
        for block in cell_genome:
            for unit in block:
                cell.append((self.allowed_ops[unit[0]], unit[1]))
                # the following lines are for NASNet search space, DARTS simply concat all nodes outputs
                # if unit[1] in cell_concat:
                #     cell_concat.remove(unit[1])
        return Genotype_cell(cell=cell, concat=cell_concat)


class DARTSEvaluator(Evaluator):
    def __init__(self,
                 categories,
                 valid_acc_model_path=get_path("valid_acc_predictor_checkpoint.json"),
                 # DARTS validation acc predictor path
                 test_acc_model_path=get_path("test_acc_predictor_checkpoint.json"),  # DARTS test acc predictor path
                 complexity_file_path=get_path("profile_data.json"),
                 # DARTS complexity file path for calc params and flops
                 objs='err&params&flops',  # objectives to be minimized
                 ):
        super().__init__(objs)

        self.feature_encoder = DARTSFeatureEncoder(categories)
        self.valid_acc_predictor = MLPPredictor(pretrained=valid_acc_model_path)
        self.test_acc_predictor = MLPPredictor(pretrained=test_acc_model_path)
        self.complexity_engine = ComplexityPredictor(complexity_file_path)

    @property
    def name(self):
        return 'DARTSEvaluator'

    def evaluate(self, archs, objs=None,
                 true_eval=False  # query the true (mean over three runs) performance
                 ):

        if objs is None:
            objs = self.objs

        features = self.feature_encoder.archs2feature(archs)
        batch_stats = []
        if true_eval:
            top1_accs = self.test_acc_predictor.predict(features)
        else:
            top1_accs = self.valid_acc_predictor.predict(features, is_noisy=True)

        for i, (arch, top1) in enumerate(zip(archs, top1_accs)):
            stats = OrderedDict()

            if 'err' in objs:
                stats['err'] = 1 - top1[0]

            if 'params' in objs:
                stats['params'] = self.complexity_engine.predict_params(arch)
            #
            if 'flops' in objs:
                stats['flops'] = self.complexity_engine.predict_macs(arch)

            batch_stats.append(stats)

        return batch_stats


class DARTSBenchmark(Benchmark):
    def __init__(self,
                 valid_acc_model_path=get_path("valid_acc_predictor_checkpoint.json"),
                 # DARTS validation acc predictor path
                 test_acc_model_path=get_path("test_acc_predictor_checkpoint.json"),  # DARTS test acc predictor path
                 complexity_file_path=get_path("profile_data.json"),
                 # DARTS complexity file path for calc params and flops
                 objs='err&params&flops',  # objectives to be minimized
                 normalized_objectives=True,  # whether to normalize the objectives
                 ):
        search_space = DARTSSearchSpace()
        evaluator = DARTSEvaluator(
            search_space.categories, valid_acc_model_path, test_acc_model_path, complexity_file_path, objs)

        super().__init__(search_space, evaluator, normalized_objectives)

    @property
    def name(self):
        return 'DARTSBenchmark'

    @property
    def _utopian_point(self):
        """ estimated from sampled architectures, use w/ caution """
        return {
            'err&params': np.array([0.05269999999999996, 392426]),
            'err&flops': np.array([0.05269999999999996, 79733440]),
            'err&params&flops': np.array([0.05269999999999996, 392426, 79733440])
        }[self.evaluator.objs]

    @property
    def _nadir_point(self):
        """ estimated from sampled architectures, use w/ caution """
        return {
            'err&params': np.array([0.275, 1672426]),
            'err&flops': np.array([0.275, 270342848]),
            'err&params&flops': np.array([0.275, 1672426, 270342848])
        }[self.evaluator.objs]

    def debug(self):
        archs = self.search_space.sample(10)
        X = self.search_space.encode(archs)
        F = self.evaluate(X, true_eval=False)
        hv = self.calc_perf_indicator(X, 'hv')

        print(archs)
        print(X)
        print(F)
        print(hv)


if __name__ == '__main__':
    benchmark = DARTSBenchmark(
        "./data/darts/valid_acc_predictor_checkpoint.json",
        "./data/darts/test_acc_predictor_checkpoint.json",
        "./data/darts/profile_data.json",
        objs='err&params',
        normalized_objectives=False
    )

    # print(benchmark.search_space.categories)
    # print(benchmark.search_space.n_var)
    # exit()
