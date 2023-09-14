import json
import copy
import os
from pathlib import Path
from collections import OrderedDict

import numpy as np

from evoxbench.modules import SearchSpace, Evaluator, Benchmark, MLPPredictor

__all__ = ['ResNet50DSearchSpace', 'ResNet50DEvaluator', 'ResNet50DBenchmark']

IMAGE_SIZE_LIST = (128, 144, 160, 176, 192, 224, 240, 256)
DEPTH_LIST = (0, 1, 2)
EXPAND_RATIO_LIST = (0.2, 0.25, 0.35)
WIDTH_MULT_LIST = (0.65, 0.8, 1.0)
BASE_DEPTH_LIST = (2, 2, 4, 2)


def get_path(name):
    return str(Path(os.environ.get("EVOXBENCH_MODEL", os.getcwd())) / "resnet" / name)


# ------------------- Following functions are specific to ResNet50D search space ------------------- #
class ResNet50DFeatureEncoder:
    def __init__(
            self,
            image_size_list=IMAGE_SIZE_LIST,
            depth_list=DEPTH_LIST,
            expand_list=EXPAND_RATIO_LIST,
            width_mult_list=WIDTH_MULT_LIST,
            base_depth_list=BASE_DEPTH_LIST,
    ):
        self.image_size_list = image_size_list
        self.expand_list = expand_list
        self.depth_list = depth_list
        self.width_mult_list = width_mult_list
        self.base_depth_list = base_depth_list

        """" build info dict """
        self.n_dim = 0
        # resolution
        self.r_info = dict(id2val={}, val2id={}, L=[], R=[])
        self._build_info_dict(target="r")
        # input stem skip
        self.input_stem_d_info = dict(id2val={}, val2id={}, L=[], R=[])
        self._build_info_dict(target="input_stem_d")
        # width_mult
        self.width_mult_info = dict(id2val=[], val2id=[], L=[], R=[])
        self._build_info_dict(target="width_mult")
        # expand ratio
        self.e_info = dict(id2val=[], val2id=[], L=[], R=[])
        self._build_info_dict(target="e")

    @property
    def n_stage(self):
        return len(self.base_depth_list)

    @property
    def max_n_blocks(self):
        return sum(self.base_depth_list) + self.n_stage * max(self.depth_list)

    def _build_info_dict(self, target):
        if target == "r":
            target_dict = self.r_info
            target_dict["L"].append(self.n_dim)
            for img_size in self.image_size_list:
                target_dict["val2id"][img_size] = self.n_dim
                target_dict["id2val"][self.n_dim] = img_size
                self.n_dim += 1
            target_dict["R"].append(self.n_dim)
        elif target == "input_stem_d":
            target_dict = self.input_stem_d_info
            target_dict["L"].append(self.n_dim)
            for skip in [0, 1]:
                target_dict["val2id"][skip] = self.n_dim
                target_dict["id2val"][self.n_dim] = skip
                self.n_dim += 1
            target_dict["R"].append(self.n_dim)
        elif target == "e":
            target_dict = self.e_info
            choices = self.expand_list
            for i in range(self.max_n_blocks):
                target_dict["val2id"].append({})
                target_dict["id2val"].append({})
                target_dict["L"].append(self.n_dim)
                for e in choices:
                    target_dict["val2id"][i][e] = self.n_dim
                    target_dict["id2val"][i][self.n_dim] = e
                    self.n_dim += 1
                target_dict["R"].append(self.n_dim)
        elif target == "width_mult":
            target_dict = self.width_mult_info
            choices = list(range(len(self.width_mult_list)))
            for i in range(self.n_stage + 2):
                target_dict["val2id"].append({})
                target_dict["id2val"].append({})
                target_dict["L"].append(self.n_dim)
                for w in choices:
                    target_dict["val2id"][i][w] = self.n_dim
                    target_dict["id2val"][i][self.n_dim] = w
                    self.n_dim += 1
                target_dict["R"].append(self.n_dim)

    def arch2feature(self, arch_dict):
        d, e, w, r = (
            arch_dict["d"],
            arch_dict["e"],
            arch_dict["w"],
            arch_dict["r"],
        )
        input_stem_skip = 1 if d[0] > 0 else 0
        d = d[1:]

        feature = np.zeros(self.n_dim)
        feature[self.r_info["val2id"][r]] = 1
        feature[self.input_stem_d_info["val2id"][input_stem_skip]] = 1
        for i in range(self.n_stage + 2):
            feature[self.width_mult_info["val2id"][i][w[i]]] = 1

        start_pt = 0
        for i, base_depth in enumerate(self.base_depth_list):
            depth = base_depth + d[i]
            for j in range(start_pt, start_pt + depth):
                feature[self.e_info["val2id"][j][e[j]]] = 1
            start_pt += max(self.depth_list) + base_depth

        return feature

    def feature2arch(self, feature):
        img_sz = self.r_info["id2val"][
            int(np.argmax(feature[self.r_info["L"][0]: self.r_info["R"][0]]))
            + self.r_info["L"][0]
            ]
        input_stem_skip = (
                self.input_stem_d_info["id2val"][
                    int(
                        np.argmax(
                            feature[
                            self.input_stem_d_info["L"][0]: self.input_stem_d_info[
                                "R"
                            ][0]
                            ]
                        )
                    )
                    + self.input_stem_d_info["L"][0]
                    ]
                * 2
        )
        assert img_sz in self.image_size_list
        arch_dict = {"d": [input_stem_skip], "e": [], "w": [], "r": img_sz}

        for i in range(self.n_stage + 2):
            arch_dict["w"].append(
                self.width_mult_info["id2val"][i][
                    int(
                        np.argmax(
                            feature[
                            self.width_mult_info["L"][i]: self.width_mult_info[
                                "R"
                            ][i]
                            ]
                        )
                    )
                    + self.width_mult_info["L"][i]
                    ]
            )

        d = 0
        skipped = 0
        stage_id = 0
        for i in range(self.max_n_blocks):
            skip = True
            for j in range(self.e_info["L"][i], self.e_info["R"][i]):
                if feature[j] == 1:
                    arch_dict["e"].append(self.e_info["id2val"][i][j])
                    skip = False
                    break
            if skip:
                arch_dict["e"].append(0)
                skipped += 1
            else:
                d += 1

            if (
                    i + 1 == self.max_n_blocks
                    or (skipped + d)
                    % (max(self.depth_list) + self.base_depth_list[stage_id])
                    == 0
            ):
                arch_dict["d"].append(d - self.base_depth_list[stage_id])
                d, skipped = 0, 0
                stage_id += 1
        return arch_dict


class FLOPsPredictor:
    channel_list = [[40, 48, 64], [168, 208, 256], [336, 408, 512], [664, 816, 1024], [1328, 1640, 2048]]
    stride_list = [1, 2, 2, 2]
    base_depth_list = [2, 2, 4, 2]

    def __init__(self, data_file_path=get_path("flops_model.json")):
        with open(data_file_path, 'r') as f:
            self.flops_dict = json.load(f)

    def predict_flops(self, sample):
        input_size = sample.get("r", 224)

        assert "d" in sample and "e" in sample and "w" in sample
        assert len(sample["d"]) == 5
        assert len(sample["e"]) == 18
        assert len(sample["w"]) == 6
        assert str(input_size) in self.flops_dict
        flops_dict = self.flops_dict[str(input_size)]

        flops = 0
        # calculate the input_stem

        flops = flops + flops_dict["input_stem"][str(sample['w'][0])][str(sample['w'][1])][str(sample['d'][0])]
        # calculate the  blocks
        base_depth = 0
        for block_id in range(4):

            for i in range(self.base_depth_list[block_id] + sample['d'][block_id + 1]):
                idx = base_depth + i

                if i == 0:
                    stride = self.stride_list[block_id]
                    in_channel = self.channel_list[block_id][sample['w'][block_id + 1]]
                    type = "head"
                else:
                    stride = 1
                    in_channel = self.channel_list[block_id + 1][sample['w'][block_id + 2]]
                    type = "follow"
                out_channel = self.channel_list[block_id + 1][sample['w'][block_id + 2]]
                expand_ratio = sample['e'][idx]

                flops = flops + flops_dict['ResNetBottleneckBlock'][str(block_id)][str(in_channel)][
                    str(out_channel)][str(stride)][str(expand_ratio)][type]

            base_depth = base_depth + self.base_depth_list[block_id] + 2

        # calculate the linear layer
        in_channel = self.channel_list[-1][sample['w'][-1]]
        flops = flops + flops_dict["LinearLayer"][str(in_channel)]

        return flops


class ParamsPredictor:
    channel_list = [[40, 48, 64], [168, 208, 256], [336, 408, 512], [664, 816, 1024], [1328, 1640, 2048]]
    stride_list = [1, 2, 2, 2]
    base_depth_list = [2, 2, 4, 2]

    def __init__(self, data_file_path=get_path("params_model.json")):
        with open(data_file_path, 'r') as f:
            self.params_dict = json.load(f)

    def predict_params(self, sample):
        assert "d" in sample and "e" in sample and "w" in sample
        assert len(sample["d"]) == 5
        assert len(sample["e"]) == 18
        assert len(sample["w"]) == 6

        params = 0
        # calculate the input_stem

        params = params + self.params_dict["input_stem"][str(sample['w'][0])][str(sample['w'][1])][str(sample['d'][0])]
        # calculate the  blocks
        base_depth = 0
        for block_id in range(4):

            for i in range(self.base_depth_list[block_id] + sample['d'][block_id + 1]):
                idx = base_depth + i

                if i == 0:
                    stride = self.stride_list[block_id]
                    in_channel = self.channel_list[block_id][sample['w'][block_id + 1]]
                else:
                    stride = 1
                    in_channel = self.channel_list[block_id + 1][sample['w'][block_id + 2]]
                out_channel = self.channel_list[block_id + 1][sample['w'][block_id + 2]]
                expand_ratio = sample['e'][idx]

                params = params + self.params_dict['ResNetBottleneckBlock'][str(block_id)][str(in_channel)][
                    str(out_channel)][str(stride)][str(expand_ratio)]

            base_depth = base_depth + self.base_depth_list[block_id] + 2

        # calculate the linear layer
        in_channel = self.channel_list[-1][sample['w'][-1]]
        params = params + self.params_dict["LinearLayer"][str(in_channel)]

        return params


class ResNet50DSearchSpace(SearchSpace):

    def __init__(self,
                 image_scale=IMAGE_SIZE_LIST,  # (min img size, max img size)
                 depth_list=DEPTH_LIST,  # depth
                 expand_ratio_list=EXPAND_RATIO_LIST,  # expansion ratio
                 width_mult_list=WIDTH_MULT_LIST,  # width multiplier
                 base_depth_list=BASE_DEPTH_LIST,  # base depth list
                 **kwargs):

        super().__init__(**kwargs)

        self.image_scale_list = list(image_scale)
        self.depth_list = list(depth_list)
        self.expand_ratio_list = list(expand_ratio_list)
        self.width_mult_list = list(width_mult_list)
        self.base_depth_list = base_depth_list

        # upper and lower bound on the decision variables
        self.n_var = 25
        # [image resolution, stem_width1, stem_width2, layer 1, layer 2, ..., layer 18]
        # image resolution ~ [128, 136, 144, ..., 256]
        # the last two layers in each stage can be skipped
        self.lb = [0] + [0, 0, 1, 1, 1, 1] + [1, 1, 0, 0] + [1, 1, 0, 0] + [1, 1, 1, 1, 0, 0] + [1, 1, 0, 0]
        self.ub = [len(self.image_scale_list) - 1] + [len(width_mult_list)] * 6 + [len(expand_ratio_list)] * 18

        # create the categories for each variable
        self.categories = [list(range(a, b + 1)) for a, b in zip(self.lb, self.ub)]

        # assuming maximum 4 layers per stage for 5 stages
        self.stage_width_indices = [1, 2, 3, 4, 5, 6]
        self.stage_layer_indices = [[7, 8, 9, 10], [11, 12, 13, 14], [15, 16, 17, 18, 19, 20], [21, 22, 23, 24]]

    @property
    def name(self):
        return 'ResNet50DSearchSpace'

    def _sample(self, phenotype=True):

        # uniform random sampling
        x = np.array([np.random.choice(options) for options in self.categories]).astype(int)

        # repair, in case of skipping the last second but not the last layer in a stage
        for indices in self.stage_layer_indices:
            if x[indices[-2]] == 0 and x[indices[-1]] > 0:
                x[indices[-2]] = x[indices[-1]]
                x[indices[-1]] = 0

        if phenotype:
            return self._decode(x)
        else:
            return x

    def _encode(self, arch):
        # a sample subnet string
        # {'r': 224,
        #  'w': [0, 0, 1, 0, 0, 1],
        #  'e': [0.35, 0.2, 0.35, 0.35, 0.25, 0.2, 0.2, 0.2, 0.35, 0.35, 0.2, 0.25, 0.2, 0.25, 0.25, 0.25, 0.2, 0.25],
        #  'd': [2, 0, 0, 2, 2]}

        x = np.array([0] * self.n_var)
        x[0] = np.where(arch['r'] == np.array(self.image_scale_list))[0][0]

        x[self.stage_width_indices] = arch['w']
        x[self.stage_width_indices] += 1

        if arch['d'][0] == 0:
            x[self.stage_width_indices[0]] = 0
            x[self.stage_width_indices[1]] = 0

        for i, (indices, d) in enumerate(zip(self.stage_layer_indices, arch['d'][1:])):
            for j in range(d + self.base_depth_list[i]):
                idx = indices[j]
                x[idx] = np.where(arch['e'][idx - 7] == np.array(self.expand_ratio_list))[0][0] + 1

        return x

    def _decode(self, x):
        # a sample decision variable vector x corresponding to the above subnet string
        # [(image scale) 3,
        #  (width mult)  2, 1, 2, 2, 2, 2
        #  (layer exp)   2, 2, 1, 1, 1, 1, 2, 0, 2, 1, 1, 1, 3, 1, 1, 2, 3, 0]
        expand_ratio_list, depth_list = [], [2]
        width_mult_list = copy.deepcopy(x[self.stage_width_indices] - 1).tolist()

        if width_mult_list[0] < 0:
            width_mult_list[0] = np.random.choice(len(self.width_mult_list))
            depth_list[0] = 0

        if width_mult_list[1] < 0:
            width_mult_list[1] = np.random.choice(len(self.width_mult_list))
            depth_list[0] = 0

        for i, indices in enumerate(self.stage_layer_indices):
            d = len(indices)
            for idx in indices:
                if x[idx] < 1:
                    d -= 1
                    e = np.random.choice(self.expand_ratio_list)
                else:
                    e = self.expand_ratio_list[x[idx] - 1]
                expand_ratio_list.append(e)
            depth_list.append(d - self.base_depth_list[i])

        return {
            'r': self.image_scale_list[x[0]],
            'w': width_mult_list, 'e': expand_ratio_list, 'd': depth_list}

    def visualize(self, arch):
        raise NotImplementedError

class ResNet50DEvaluator(Evaluator):
    def __init__(self,
                 valid_acc_model_path=get_path("valid_acc_predictor_checkpoint.json"),
                 # ResNet50 validation acc predictor path
                 test_acc_model_path=get_path("test_acc_predictor_checkpoint.json"),  # ResNet50 test acc predictor path
                 params_model_path=get_path("params_model.json"),
                 # path to pretrained Params predictor surrogate model / look-up table
                 flops_model_path=get_path("flops_model.json"),
                 # path to pretrained FLOPs predictor surrogate model / look-up table
                 objs='err&flops&latency',  # objectives to be minimized
                 ):
        super().__init__(objs)

        self.feature_encoder = ResNet50DFeatureEncoder()
        self.valid_acc_predictor = MLPPredictor(pretrained=valid_acc_model_path)
        self.test_acc_predictor = MLPPredictor(pretrained=test_acc_model_path)

        self.params_predictor = ParamsPredictor(params_model_path)
        self.flops_predictor = FLOPsPredictor(flops_model_path)

        # # prepare meta file for calculating statistics
        # self.pf = np.array(json.load(open(pf_file_path, 'r'))[str(fidelity)][objs])

    @property
    def name(self):
        return 'ResNet50DEvaluator'

    def evaluate(self, archs, objs=None,
                 true_eval=False  # query the true (mean over three runs) performance
                 ):

        if objs is None:
            objs = self.objs

        features = np.array([self.feature_encoder.arch2feature(arch) for arch in archs])

        batch_stats = []

        if true_eval:
            top1_accs = self.test_acc_predictor.predict(features)
        else:
            top1_accs = self.valid_acc_predictor.predict(features, is_noisy=True)

        for i, (arch, top1) in enumerate(zip(archs, top1_accs)):
            stats = OrderedDict()

            # todo: add stochastic evaluation method
            if 'err' in objs:
                stats['err'] = 1 - top1[0]

            if 'params' in objs:
                stats['params'] = self.params_predictor.predict_params(arch)

            if 'flops' in objs:
                stats['flops'] = self.flops_predictor.predict_flops(arch)

            batch_stats.append(stats)

        return batch_stats


class ResNet50DBenchmark(Benchmark):
    def __init__(self,
                 valid_acc_model_path=get_path("valid_acc_predictor_checkpoint.json"),
                 # ResNet50 validation acc predictor path
                 test_acc_model_path=get_path("test_acc_predictor_checkpoint.json"),  # ResNet50 test acc predictor path
                 params_model_path=get_path("params_model.json"),
                 # path to pretrained Params predictor surrogate model / look-up table
                 flops_model_path=get_path("flops_model.json"),
                 # path to pretrained FLOPs predictor surrogate model / look-up table
                 objs='err&flops&latency',  # objectives to be minimized
                 normalized_objectives=False,  # whether to normalize the objectives
                 ):
        search_space = ResNet50DSearchSpace()
        evaluator = ResNet50DEvaluator(
            valid_acc_model_path, test_acc_model_path, params_model_path, flops_model_path, objs)
        super().__init__(search_space, evaluator, normalized_objectives)

    @property
    def name(self):
        return 'ResNet50DBenchmark'

    @property
    def _utopian_point(self):
        """ estimated from sampled architectures, use w/ caution """
        return {
            'err&params': [1.987000e-01, 6.542624e+06],
            'err&flops': [1.98700000e-01, 6.69461472e+08],
            'err&params&flops': [1.98700000e-01, 6.54262400e+06, 6.69461472e+08],
        }[self.evaluator.objs]

    @property
    def _nadir_point(self):
        """ estimated from sampled architectures, use w/ caution """
        return {
            'err&params': [3.1242000e-01, 4.4114544e+07],
            'err&flops': [3.12420000e-01, 1.45769615e+10],
            'err&params&flops': [3.12420000e-01, 4.41145440e+07, 1.45769615e+10],
        }[self.evaluator.objs]

    def debug(self):
        archs = self.search_space.sample(10)
        X = self.search_space.encode(archs)
        F = self.evaluate(X)
        hv = self.calc_perf_indicator(X, 'hv')

        print(archs)
        print(X)
        print(F)
        print(hv)


if __name__ == '__main__':
    benchmark = ResNet50DBenchmark(
        "/Users/luzhicha/Dropbox/2022/EvoXBench/python_codes/evoxbench/benchmarks/data/resnet/valid_acc_predictor_checkpoint.json",
        "/Users/luzhicha/Dropbox/2022/EvoXBench/python_codes/evoxbench/benchmarks/data/resnet/test_acc_predictor_checkpoint.json",
        "/Users/luzhicha/Dropbox/2022/EvoXBench/python_codes/evoxbench/benchmarks/data/resnet/params_model.json",
        "/Users/luzhicha/Dropbox/2022/EvoXBench/python_codes/evoxbench/benchmarks/data/resnet/flops_model.json",
        objs='err&params', normalized_objectives=True
    )

    benchmark.debug()
