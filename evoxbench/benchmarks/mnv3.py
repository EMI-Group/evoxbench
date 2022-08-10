import os
import copy
from pathlib import Path

import yaml
import json
import random
import numpy as np
from collections import OrderedDict

from evoxbench.modules import SearchSpace, Evaluator, Benchmark, MLPPredictor

__all__ = ['MobileNetV3SearchSpace', 'MobileNetV3Evaluator', 'MobileNetV3Benchmark']

IMAGE_SIZE_LIST = (160, 176, 192, 208, 224)
KERNEL_SIZE_LIST = (3, 5, 7)
DEPTH_LIST = (2, 3, 4)
EXPAND_RATIO_LIST = (3, 4, 6)
WIDTH_MULT = 1.2


def get_path(name):
    return str(Path(os.environ.get("EVOXBENCH_MODEL", os.getcwd())) / "mnv3" / name)


# ------------------- Following functions are specific to MobileNetV3 search space ------------------- #
class MobileNetV3FeatureEncoder(object):
    def __init__(self):

        self.ks_map = self.construct_maps(keys=KERNEL_SIZE_LIST)
        self.ex_map = self.construct_maps(keys=EXPAND_RATIO_LIST)
        self.dp_map = self.construct_maps(keys=DEPTH_LIST)

    # Helper for constructing the one-hot vectors.
    @staticmethod
    def construct_maps(keys):
        d = dict()
        keys = list(set(keys))
        for k in keys:
            if k not in d:
                d[k] = len(list(d.keys()))
        return d

    def arch2feature(self, arch):
        # This function converts a network config to a feature vector (128-D).
        ks_list = copy.deepcopy(arch["ks"])
        ex_list = copy.deepcopy(arch["e"])
        d_list = copy.deepcopy(arch["d"])
        r = copy.deepcopy(arch["r"])

        start = 0
        end = 4
        for d in d_list:
            for j in range(start + d, end):
                ks_list[j] = 0
                ex_list[j] = 0
            start += 4
            end += 4

        # convert to onehot
        ks_onehot = [0 for _ in range(60)]
        ex_onehot = [0 for _ in range(60)]
        r_onehot = [0 for _ in range(8)]

        for i in range(20):
            start = i * 3
            if ks_list[i] != 0:
                ks_onehot[start + self.ks_map[ks_list[i]]] = 1
            if ex_list[i] != 0:
                ex_onehot[start + self.ex_map[ex_list[i]]] = 1

        r_onehot[(r - 112) // 16] = 1
        return ks_onehot + ex_onehot + r_onehot


class FLOPsPredictor:
    channel_list = [24, 32, 48, 96, 136, 192]  # WIDTH_MULT = 1.2
    stride_stages = [2, 2, 2, 1, 2]
    act_stages = ["relu", "relu", "h_swish", "h_swish", "h_swish"]
    se_stages = ['false', 'true', 'false', 'true', 'true']

    def __init__(self,
                 data_file_path=get_path("flops_model.json")):
        with open(data_file_path, 'r') as f:
            self.flops_dict = json.load(f)

    def predict_flops(self, sample):
        input_size = sample.get("r", 224)
        flops_dict = self.flops_dict[str(input_size)]
        assert "ks" in sample and "e" in sample and "d" in sample
        assert len(sample["ks"]) == len(sample["e"]) and len(sample["ks"]) == 20
        assert len(sample["d"]) == 5

        flops = 0
        # calculate the dynamic blocks
        size = input_size // 2
        for block_id in range(5):
            for i in range(sample['d'][block_id]):
                idx = block_id * 4 + i
                out_channel = self.channel_list[block_id + 1]
                k = sample['ks'][idx]
                e = sample['e'][idx]
                if i == 0:
                    s = self.stride_stages[block_id]
                    in_channel = self.channel_list[block_id]
                else:
                    s = 1
                    in_channel = self.channel_list[block_id + 1]
                act = self.act_stages[block_id]
                se = self.se_stages[block_id]

                in_channel = str(in_channel)
                out_channel = str(out_channel)
                k = str(k)
                e = str(e)
                s = str(s)
                size_str = str(size)
                # print(in_channel,out_channel,size, k,e,s,act,se)
                flops = flops + flops_dict['MBConvLayerBlock'][in_channel][out_channel][size_str][k][e][s][act][se]
                if s == "2":  # stride = 2 means half the HW
                    size = (size + 1) // 2

        flops += flops_dict["FirstBlockLayer"]
        flops += flops_dict["LastLayers"]
        return flops


class ParamsPredictor:
    channel_list = [24, 32, 48, 96, 136, 192]  # WIDTH_MULT = 1.2
    stride_stages = [2, 2, 2, 1, 2]
    act_stages = ["relu", "relu", "h_swish", "h_swish", "h_swish"]
    se_stages = ['false', 'true', 'false', 'true', 'true']

    def __init__(self,
                 data_file_path=get_path("params_model.json")):
        with open(data_file_path, 'r') as f:
            self.params_dict = json.load(f)

    def predict_params(self, sample):
        assert "ks" in sample and "e" in sample and "d" in sample
        assert len(sample["ks"]) == len(sample["e"]) and len(sample["ks"]) == 20
        assert len(sample["d"]) == 5

        params = 0
        # calculate the dynamic layers

        for block_id in range(5):
            for i in range(sample['d'][block_id]):
                idx = block_id * 4 + i
                # print(idx)
                out_channel = self.channel_list[block_id + 1]
                k = sample['ks'][idx]
                e = sample['e'][idx]
                if i == 0:
                    s = self.stride_stages[block_id]
                    in_channel = self.channel_list[block_id]
                else:
                    s = 1
                    in_channel = self.channel_list[block_id + 1]
                act = self.act_stages[block_id]
                se = self.se_stages[block_id]
                in_channel = str(in_channel)
                out_channel = str(out_channel)
                k = str(k)
                e = str(e)
                s = str(s)
                # print(in_channel, out_channel, k, e, s, act, se)
                params = params + self.params_dict['MBConvLayer'][in_channel][out_channel][k][e][s][act][se]
        params = params + self.params_dict['OtherLayer']

        return params


class LatencyEstimator(object):
    def __init__(self, yaml_file_path=''):

        with open(yaml_file_path, "r") as fp:
            self.lut = yaml.load(fp, Loader=yaml.FullLoader)

    @staticmethod
    def repr_shape(shape):
        if isinstance(shape, (list, tuple)):
            return "x".join(str(_) for _ in shape)
        elif isinstance(shape, str):
            return shape
        else:
            return TypeError

    def query(
            self,
            l_type: str,
            input_shape,
            output_shape,
            mid=None,
            ks=None,
            stride=None,
            id_skip=None,
            se=None,
            h_swish=None,
    ):
        infos = [
            l_type,
            "input:%s" % self.repr_shape(input_shape),
            "output:%s" % self.repr_shape(output_shape),
        ]

        if l_type in ("expanded_conv",):
            assert None not in (mid, ks, stride, id_skip, se, h_swish)
            infos += [
                "expand:%d" % mid,
                "kernel:%d" % ks,
                "stride:%d" % stride,
                "idskip:%d" % id_skip,
                "se:%d" % se,
                "hs:%d" % h_swish,
            ]
        key = "-".join(infos)
        return self.lut[key]["mean"]

    def predict_network_latency(self, net, image_size=224):
        predicted_latency = 0
        # first conv
        predicted_latency += self.query(
            "Conv",
            [image_size, image_size, 3],
            [(image_size + 1) // 2, (image_size + 1) // 2, net.first_conv.out_channels],
        )
        # blocks
        fsize = (image_size + 1) // 2
        for block in net.blocks:
            mb_conv = block.mobile_inverted_conv
            shortcut = block.shortcut

            if mb_conv is None:
                continue
            if shortcut is None:
                idskip = 0
            else:
                idskip = 1
            out_fz = int((fsize - 1) / mb_conv.stride + 1)
            block_latency = self.query(
                "expanded_conv",
                [fsize, fsize, mb_conv.in_channels],
                [out_fz, out_fz, mb_conv.out_channels],
                mid=mb_conv.depth_conv.conv.in_channels,
                ks=mb_conv.kernel_size,
                stride=mb_conv.stride,
                id_skip=idskip,
                se=1 if mb_conv.use_se else 0,
                h_swish=1 if mb_conv.act_func == "h_swish" else 0,
            )
            predicted_latency += block_latency
            fsize = out_fz
        # final expand layer
        predicted_latency += self.query(
            "Conv_1",
            [fsize, fsize, net.final_expand_layer.in_channels],
            [fsize, fsize, net.final_expand_layer.out_channels],
        )
        # global average pooling
        predicted_latency += self.query(
            "AvgPool2D",
            [fsize, fsize, net.final_expand_layer.out_channels],
            [1, 1, net.final_expand_layer.out_channels],
        )
        # feature mix layer
        predicted_latency += self.query(
            "Conv_2",
            [1, 1, net.feature_mix_layer.in_channels],
            [1, 1, net.feature_mix_layer.out_channels],
        )
        # classifier
        predicted_latency += self.query(
            "Logits", [1, 1, net.classifier.in_features], [net.classifier.out_features]
        )
        return predicted_latency

    def predict_network_latency_given_spec(self, spec):
        image_size = spec["r"]
        predicted_latency = 0
        # first conv
        predicted_latency += self.query(
            "Conv",
            [image_size, image_size, 3],
            [(image_size + 1) // 2, (image_size + 1) // 2, 24],
        )
        # blocks
        fsize = (image_size + 1) // 2
        # first block
        predicted_latency += self.query(
            "expanded_conv",
            [fsize, fsize, 24],
            [fsize, fsize, 24],
            mid=24,
            ks=3,
            stride=1,
            id_skip=1,
            se=0,
            h_swish=0,
        )
        in_channel = 24
        stride_stages = [2, 2, 2, 1, 2]
        width_stages = [32, 48, 96, 136, 192]
        act_stages = ["relu", "relu", "h_swish", "h_swish", "h_swish"]
        se_stages = [False, True, False, True, True]
        for i in range(20):
            stage = i // 4
            depth_max = spec["d"][stage]
            depth = i % 4 + 1
            if depth > depth_max:
                continue
            ks, e = spec["ks"][i], spec["e"][i]
            if i % 4 == 0:
                stride = stride_stages[stage]
                idskip = 0
            else:
                stride = 1
                idskip = 1
            out_channel = width_stages[stage]
            out_fz = int((fsize - 1) / stride + 1)

            mid_channel = round(in_channel * e)
            block_latency = self.query(
                "expanded_conv",
                [fsize, fsize, in_channel],
                [out_fz, out_fz, out_channel],
                mid=mid_channel,
                ks=ks,
                stride=stride,
                id_skip=idskip,
                se=1 if se_stages[stage] else 0,
                h_swish=1 if act_stages[stage] == "h_swish" else 0,
            )
            predicted_latency += block_latency
            fsize = out_fz
            in_channel = out_channel
        # final expand layer
        predicted_latency += self.query(
            "Conv_1",
            [fsize, fsize, 192],
            [fsize, fsize, 1152],
        )
        # global average pooling
        predicted_latency += self.query(
            "AvgPool2D",
            [fsize, fsize, 1152],
            [1, 1, 1152],
        )
        # feature mix layer
        predicted_latency += self.query("Conv_2", [1, 1, 1152], [1, 1, 1536])
        # classifier
        predicted_latency += self.query("Logits", [1, 1, 1536], [1000])
        return predicted_latency


class LatencyPredictor:
    def __init__(self, data_root=get_path("note10"),
                 resolutions=(160, 176, 192, 208, 224)):
        self.latency_tables = {}
        # device = os.path.basename(data_root)

        for image_size in resolutions:
            self.latency_tables[image_size] = LatencyEstimator(
                os.path.join(data_root, '{}_lookup_table.yaml'.format(image_size)))
            # print("Built latency table for image size: %d." % image_size)

    def predict_latency(self, spec: dict):
        return self.latency_tables[spec["r"]].predict_network_latency_given_spec(
            spec
        )


class MobileNetV3SearchSpace(SearchSpace):

    def __init__(self,
                 image_scale=IMAGE_SIZE_LIST,  # (min img size, max img size)
                 ks_list=KERNEL_SIZE_LIST,  # kernel sizes
                 depth_list=DEPTH_LIST,  # depth
                 expand_ratio_list=EXPAND_RATIO_LIST,  # expansion ratio
                 width_mult=WIDTH_MULT,  # width multiplier
                 **kwargs):

        super().__init__(**kwargs)

        self.image_scale_list = list(image_scale)
        self.ks_list = list(ks_list)
        self.depth_list = list(depth_list)
        self.expand_ratio_list = list(expand_ratio_list)
        self.width_mult = width_mult

        # upper and lower bound on the decision variables
        self.n_var = 21
        # [image resolution, layer 1, layer 2, ..., layer 20]
        # image resolution ~ [128, 136, 144, ..., 224]
        # the last two layers in each stage can be skipped
        self.lb = [0] + [1, 1, 0, 0] + [1, 1, 0, 0] + [1, 1, 0, 0] + [1, 1, 0, 0] + [1, 1, 0, 0]
        self.ub = [len(self.image_scale_list) - 1]

        if max(depth_list) == 4:
            self.ub += [int(len(ks_list) * len(expand_ratio_list))] * 20
        elif max(depth_list) == 3:
            self.ub += ([int(len(ks_list) * len(expand_ratio_list))] * 3 + [0]) * 5
        elif max(depth_list) == 2:
            self.ub += ([int(len(ks_list) * len(expand_ratio_list))] * 2 + [0, 0]) * 5
        else:
            ValueError("max depth has to be greater than 2")

        # create the categories for each variable
        self.categories = [list(range(a, b + 1)) for a, b in zip(self.lb, self.ub)]

        # assuming maximum 4 layers per stage for 5 stages
        self.stage_layer_indices = [list(range(1, self.n_var))[i:i + 4]
                                    for i in range(0, 20, 4)]

        # create the mappings between decision variables (genotype) and subnet architectural string (phenotype)
        self.str2var_mapping = OrderedDict()
        self.var2str_mapping = OrderedDict()
        self.str2var_mapping['skip'] = 0
        increment = 1
        for e in self.expand_ratio_list:
            for ks in self.ks_list:
                self.str2var_mapping['ks@{}_e@{}'.format(ks, e)] = increment
                self.var2str_mapping[increment] = (ks, e)
                increment += 1

    @property
    def name(self):
        return 'MobileNetV3SearchSpace'

    def str2var(self, ks, e):
        return self.str2var_mapping['ks@{}_e@{}'.format(ks, e)]

    def var2str(self, v, ub):
        if v > 0:
            return self.var2str_mapping[v]
        else:
            return self.var2str_mapping[random.randint(1, max(ub, 1))]

    def _sample(self, phenotype=True):

        # uniform random sampling
        x = [random.choice(options) for options in self.categories]

        x = np.array(x).astype(int)

        # repair, in case of skipping the third but not the fourth layer in a stage
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
        # {'r' : 224,
        #  'ks': [7, 7, 7, 7, 7, 3, 5, 3, 3, 5, 7, 3, 5, 5, 3, 3, 3, 3, 3, 5],
        #  'e' : [4, 6, 4, 6, 6, 6, 6, 6, 3, 4, 4, 4, 6, 4, 4, 3, 3, 6, 3, 4],
        #  'd' : [2, 2, 3, 4, 2]}

        x = [0] * self.n_var
        x[0] = np.where(arch['r'] == np.array(self.image_scale_list))[0][0]

        for indices, d in zip(self.stage_layer_indices, arch['d']):
            for i in range(d):
                idx = indices[i]
                x[idx] = self.str2var(arch['ks'][idx - 1], arch['e'][idx - 1])
        return x

    def _decode(self, x):
        # a sample decision variable vector x corresponding to the above subnet string
        # [(image scale) 4,
        #  (layers)      8, 9, 5, 5, 6, 2, 3, 6, 6, 1, 4, 0, 1, 2, 2, 3, 9, 5, 8, 1]

        ks_list, expand_ratio_list, depth_list = [], [], []
        for indices in self.stage_layer_indices:
            d = len(indices)
            for idx in indices:
                ks, e = self.var2str(x[idx], self.ub[idx])
                ks_list.append(ks)
                expand_ratio_list.append(e)
                if x[idx] < 1:
                    d -= 1
            depth_list.append(d)

        return {
            'r': self.image_scale_list[x[0]],
            'ks': ks_list, 'e': expand_ratio_list, 'd': depth_list}

    def visualize(self, arch):
        raise NotImplementedError


class MobileNetV3Evaluator(Evaluator):
    def __init__(self,
                 valid_acc_model_path=get_path("valid_acc_predictor_checkpoint.json"),
                 # MNV3 validation acc predictor path
                 test_acc_model_path=get_path("test_acc_predictor_checkpoint.json"),
                 # MNV3 test acc predictor path
                 params_model_path=get_path("params_model.json"),
                 # path to pretrained Params predictor surrogate model / look-up table
                 flops_model_path=get_path("flops_model.json"),
                 # path to pretrained FLOPs predictor surrogate model / look-up table
                 latency_model_path=get_path("note10"),
                 # path to pretrained latency predictor surrogate model / look-up table
                 objs='err&flops&latency',  # objectives to be minimized
                 ):
        super().__init__(objs)

        self.feature_encoder = MobileNetV3FeatureEncoder()
        self.valid_acc_predictor = MLPPredictor(pretrained=valid_acc_model_path)
        self.test_acc_predictor = MLPPredictor(pretrained=test_acc_model_path)

        self.params_predictor = ParamsPredictor(params_model_path)
        self.flops_predictor = FLOPsPredictor(flops_model_path)
        self.latency_predictor = LatencyPredictor(latency_model_path)

    @property
    def name(self):
        return 'MobileNetV3Evaluator'

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

            if 'latency' in objs:
                stats['latency'] = self.latency_predictor.predict_latency(arch)  # in ms

            batch_stats.append(stats)
        return batch_stats


class MobileNetV3Benchmark(Benchmark):
    def __init__(self,
                 valid_acc_model_path=get_path("valid_acc_predictor_checkpoint.json"),
                 # MNV3 validation acc predictor path
                 test_acc_model_path=get_path("test_acc_predictor_checkpoint.json"),
                 # MNV3 test acc predictor path
                 params_model_path=get_path("params_model.json"),
                 # path to pretrained Params predictor surrogate model / look-up table
                 flops_model_path=get_path("flops_model.json"),
                 # path to pretrained FLOPs predictor surrogate model / look-up table
                 latency_model_path=get_path("note10"),
                 # path to pretrained latency predictor surrogate model / look-up table
                 objs='err&flops&latency',  # objectives to be minimized
                 normalized_objectives=False,  # whether to normalize the objectives
                 ):
        search_space = MobileNetV3SearchSpace()
        evaluator = MobileNetV3Evaluator(valid_acc_model_path, test_acc_model_path,
                                         params_model_path, flops_model_path, latency_model_path, objs)
        super().__init__(search_space, evaluator, normalized_objectives)

    @property
    def name(self):
        return 'MobileNetV3Benchmark'

    @property
    def _utopian_point(self):
        """ estimated from sampled architectures, use w/ caution """
        return {
            'err&params': [2.091800e-01, 4.610136e+06],
            'err&flops': [2.09180000e-01, 2.13223824e+08],
            'err&latency': [0.20918, 10.35887161],
            'err&params&flops': [2.09180000e-01, 4.61013600e+06, 2.13223824e+08],
            'err&params&latency': [2.09180000e-01, 4.61013600e+06, 1.03588716e+01],
            'err&flops&latency': [2.09180000e-01, 2.13223824e+08, 1.03588716e+01],
            'err&params&flops&latency': [2.09180000e-01, 4.61013600e+06, 2.13223824e+08, 1.03588716e+01],
        }[self.evaluator.objs]

    @property
    def _nadir_point(self):
        """ estimated from sampled architectures, use w/ caution """
        return {
            'err&params': [2.9796000e-01, 1.0197992e+07],
            'err&flops': [2.97960000e-01, 1.37679763e+09],
            'err&latency': [0.29796, 70.38058926],
            'err&params&flops': [2.97960000e-01, 1.01979920e+07, 1.37679763e+09],
            'err&params&latency': [2.97960000e-01, 1.01979920e+07, 7.03805893e+01],
            'err&flops&latency': [2.97960000e-01, 1.37679763e+09, 7.03805893e+01],
            'err&params&flops&latency': [2.97960000e-01, 1.01979920e+07, 1.37679763e+09, 7.03805893e+01],
        }[self.evaluator.objs]

    def debug(self):
        archs = self.search_space.sample(10)
        X = self.search_space.encode(archs)
        F = self.evaluate(X, true_eval=True)
        hv = self.calc_perf_indicator(X, 'hv')

        print(archs)
        print(X)
        print(F)
        print(hv)


if __name__ == '__main__':
    benchmark = MobileNetV3Benchmark(
        "/Users/luzhicha/Dropbox/2022/EvoXBench/python_codes/evoxbench/benchmarks/data/mnv3/valid_acc_predictor_checkpoint.json",
        "/Users/luzhicha/Dropbox/2022/EvoXBench/python_codes/evoxbench/benchmarks/data/mnv3/test_acc_predictor_checkpoint.json",
        "/Users/luzhicha/Dropbox/2022/EvoXBench/python_codes/evoxbench/benchmarks/data/mnv3/params_model.json",
        "/Users/luzhicha/Dropbox/2022/EvoXBench/python_codes/evoxbench/benchmarks/data/mnv3/flops_model.json",
        "/Users/luzhicha/Dropbox/2022/EvoXBench/python_codes/evoxbench/benchmarks/data/mnv3/note10",
        objs='err&params&flops&latency', normalized_objectives=False,
    )

    # print(benchmark.search_space.n_var)
    # exit()

    benchmark.debug()
