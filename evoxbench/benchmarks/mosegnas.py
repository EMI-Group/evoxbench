import os
import random
import json
import math
from pathlib import Path
import numpy as np
import itertools

from evoxbench.modules import SearchSpace, Evaluator, Benchmark, SurrogateModel

LOWER_BOUND = -1
UPPER_BOUND = 1
MIN_VALUE_OF_DATASET = [1.5595920924939386, -3.270169119255751]
MAX_VALUE_OF_DATASET = [2.206984236920299, -0.3437587186721861]


# from mosegnas.models import MoSegNASResult  # has to be imported after the init method

__all__ = [
    "MoSegNASSearchSpace",
    "MoSegNASEvaluator",
    "MoSegNASBenchmark",
    "MoSegNASSurrogateModel",
]

# HASH = {'conv1x1-compression': 0, 'conv3x3': 1, 'conv1x1-expansion': 2}


def get_path(name):
    return str(Path(os.environ.get("EVOXBENCH_MODEL", os.getcwd())) / "moseg" / name)


class MoSegNASSearchSpace(SearchSpace):
    def __init__(self, subnet_str=True, **kwargs):
        super().__init__(**kwargs)
        self.subnet_str = subnet_str
        # number of MAX layers of each stage:
        # range of number of each layer estimated
        # [0/2,
        # 0~2,
        # 0~2,
        # 0~2,
        # 0~2
        # ]
        self.depth_list = [2, 2, 3, 4, 2]

        # choice of the layer except input stem
        self.expand_ratio_list = [0.2, 0.25, 0.35]
        self.width_list = [2, 2, 2, 2, 2, 2]

        # d e w
        self.categories = [list(range(d + 1)) for d in self.depth_list]
        self.categories += [list(range(3))] * 13
        self.categories += [list(range(3))] * 6
        self.category_mapping = {
            i: {val: j for j, val in enumerate(cat)}
            for i, cat in enumerate(self.categories)
        }

    @property
    def name(self):
        return "MoSegNASSearchSpace"

    def _sample(self):
        x = np.array([random.choice(options) for options in self.categories])
        if self.subnet_str:
            return self._decode(x)
        else:
            return x

    def _encode(self, subnet_str):
        e = [
            np.where(_e == np.array(self.expand_ratio_list))[0][0]
            for _e in subnet_str["e"]
        ]
        return subnet_str["d"] + e + subnet_str["w"]

    def _decode(self, x):
        e = [self.expand_ratio_list[i] for i in x[4:-5]]
        return {"d": x[:4].tolist(), "e": e, "w": x[-5:].tolist()}

    def _one_hot_encode(self, X):
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        encoded_sample = []
        for feature_index, feature_value in enumerate(X):
            encoding = [0] * len(self.categories[feature_index])
            encoding[self.category_mapping[feature_index][feature_value]] = 1
            encoded_sample.extend(encoding)

        return encoded_sample

    def visualize(self):
        """method to visualize an architecture"""
        raise NotImplementedError


class MoSegNASBenchmark(Benchmark):
    def __init__(
        self,
        normalized_objectives=False,
        surrogate_pretrained_list={
            "latency": get_path("pretrained/surrogate_model/ranknet_latency.json"),
            "mIoU": get_path("pretrained/surrogate_model/ranknet_mIoU.json"),
        },
        pretrained_json=get_path(
            "pretrained/ofa_fanet_plus_bottleneck_rtx_fps@0.5.json"
        ),
        **kwargs
    ):
        self.search_space = MoSegNASSearchSpace()
        self.surrogate_pretrained_list = surrogate_pretrained_list
        self.pretrained_json = pretrained_json
        self.evaluator = MoSegNASEvaluator(surrogate_pretrained_list, pretrained_json)
        super().__init__(
            self.search_space, self.evaluator, normalized_objectives, **kwargs
        )

    @property
    def name(self):
        return "MoSegNASBenchmark"

    def debug(self, samples=10):
        archs = self.search_space.sample(samples)
        X = self.search_space.encode(archs)
        F = self.evaluator.evaluate(X, true_eval=True)

        print(archs)
        print(X)
        print(F)

    @property
    def _utopian_point(self):
        """ estimated from sampled architectures, use w/ caution """
        return {
        }[self.evaluator.objs]

    @property
    def _nadir_point(self):
        """ estimated from sampled architectures, use w/ caution """
        return {
        }[self.evaluator.objs]


class MoSegNASEvaluator(Evaluator):
    def __init__(self, surrogate_pretrained_list=None, pretrained_json=None, **kwargs):
        super().__init__(**kwargs)
        self.surrogate_pretrained_list = surrogate_pretrained_list

        self.feature_encoder = MoSegNASSearchSpace()
        self.surrogate_model = MoSegNASSurrogateModel(
            pretrained_weights=self.surrogate_pretrained_list,
            categories=self.feature_encoder.categories,
            lookup_table=pretrained_json,
        )

    @property
    def name(self):
        return "MoSegNASEvaluator"

    def evaluate(
        self,
        archs,  # archs is subnets
        true_eval=False,  # true_eval: if evaluate based on data or true inference result
        objs="params&flops&latency&FPS&mIoU",  # objectives to be minimized/maximized
        **kwargs
    ):
        """evalute the performance of the given subnets"""
        batch_stats = []

        for index, subnet_encoded in enumerate(archs):
            print(
                "evaluating subnet index {}, subnet {}:".format(index, subnet_encoded)
            )

            pred = self.surrogate_model.predict(
                subnet=subnet_encoded, true_eval=true_eval, objs=objs
            )
            if "FPS" in objs:
                pred["FPS"] = 1000.0 / pred["latency"]
            batch_stats.append(pred)

        return batch_stats


class MosegNASRankNet:
    def __init__(
        self,
        pretrained=None,
        n_hidden_layers=2,
        n_hidden_neurons=400,
        n_output=1,
        drop=0.2,
        trn_split=0.8,
        lr=8e-4,
        epochs=300,
        loss="mse",
    ):
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_neurons = n_hidden_neurons
        self.n_output = n_output
        self.drop = drop
        self.trn_split = trn_split
        self.lr = lr
        self.epochs = epochs
        self.n_feature = None
        self.loss = loss

        self.pretrained = pretrained
        self.weights = []
        self.biases = []
        if pretrained is not None:
            self.init_weights()
        else:
            self.randomly_init_weights()
        self.name = "RankNet"

    def init_weights(self):
        for i in range(
            int(list(self.pretrained.keys())[0][1:]),
            len(self.pretrained) // 2 + int(list(self.pretrained.keys())[0][1:]),
        ):
            self.weights.append(self.pretrained["W" + str(i)])
            self.biases.append(self.pretrained["b" + str(i)])

    def randomly_init_weights(self, x):
        self.n_feature = x.shape[1]

        # Input layer
        self.weights.append(
            [self.fill(self.n_hidden_neurons) for _ in range(self.n_feature)]
        )
        self.biases.append(self.fill(self.n_hidden_neurons))

        # Hidden layers
        for _ in range(self.n_layers):
            self.weights.append(
                [self.fill(self.n_hidden_neurons) for _ in range(self.n_hidden_neurons)]
            )
            self.biases.append(self.fill(self.n_hidden_neurons))

        # Output layer
        self.weights.append(
            [self.fill(self.n_output) for _ in range(self.n_hidden_neurons)]
        )
        self.biases.append(self.fill(self.n_output))

    @staticmethod
    def fill(x):
        return [random.uniform(-1, 1) for _ in range(x)]

    def predict(self, x):
        if x.ndim < 2:
            data = np.zeros((1, x.shape[0]), dtype=np.float32)
            data[0, :] = x
        else:
            data = x.astype(np.float32)
        data = data.T
        pred = self.forward(data)
        return pred[:, 0]

    def relu(self, x):
        return max(0, x)

    def dropout(self, x):
        return 0.0 if random.random() < self.drop else x

    def linear(self, inputs, weights, biases):
        return [
            sum(x * w for x, w in zip(inputs, weights_row)) + b
            for weights_row, b in zip(weights, biases)
        ]

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def forward(self, x):
        # Input layer
        outputs = self.linear(x, self.weights[0], self.biases[0])
        outputs = [self.relu(x) for x in outputs]

        # Hidden layers
        for layer in range(self.n_hidden_layers):
            outputs = self.linear(outputs, self.weights[layer], self.biases[layer])
            outputs = [self.relu(x) for x in outputs]

        # Output layer
        outputs = self.linear(outputs, self.weights[-1], self.biases[-1])

        return outputs

    def train(self):
        pass


class MoSegNASSurrogateModel(SurrogateModel):
    def __init__(
        self,
        surrogate_pretrained_list=None,
        pretrained_json=None,
        lookup_table=None,
        categories=[list(range(4))] * 24,
        **kwargs
    ):
        super().__init__()

        self.pretrained_result = pretrained_json
        self.lookup_table = lookup_table
        self.categories = categories
        self.searchSpace = MoSegNASSearchSpace()

        # 10 model
        if "latency" in surrogate_pretrained_list:
            self.latency_pretrained = surrogate_pretrained_list["latency"]
        if "mIoU" in surrogate_pretrained_list:
            self.mIoU_pretrained = surrogate_pretrained_list["mIoU"]
        else:
            self.latency_pretrained = None
            self.mIoU_pretrained = None

    def name(self):
        return "MoSegNASSurrogateModel"

    def fit(self, subnet):
        """subnet = [{'d': [...], 'e': [...], 'w': [...]}]
        pretrained result = [{'config': {'d': [...], 'e': [...], 'w': [...]}, 'params': ..., 'flops': ..., 'latency': ..., 'FPS': ..., 'mIoU': ...}, {...}, {...}
        """
        """ method to perform forward in a surrogate model from data """
        for result in self.pretrained_result:
            if "config" in result and isinstance(result["config"], dict):
                config = result["config"]
                if all(
                    key in config and config[key] == value
                    for key, value in subnet[0].items()
                ):
                    return [
                        result["params"],
                        result["flops"],
                        result["latency"],
                        result["mIoU"],
                    ]
        return None

    def addup_predictor(self, subnet):
        """method to predict performance only for flops or params from given architecture features(subnets)"""
        lookup_table = json.load(open(self.lookup_table, "r"))
        d_len = len(self.searchSpace.depth_list)
        w_len = len(self.searchSpace.width_list)
        e_len = len(self.searchSpace.categories) - d_len - w_len

        depth = subnet[:d_len]
        expand_ratio = subnet[d_len : d_len + e_len]
        expand_ratio = [self.searchSpace.expand_ratio_list[i] for i in expand_ratio]
        width = subnet[e_len + d_len :]

        params, flops = 0.0, 0.0
        for idx in range(len(depth)):
            if depth[idx] != 0:
                d = [0 for _ in range(d_len)]
                w = [0 for _ in range(w_len)]
                e = [0.0 for _ in range(e_len)]
                d[idx] = depth[idx]
                previous = sum(self.searchSpace.depth_list[0:idx])
                length = self.searchSpace.depth_list[idx]
                e[previous : previous + length] = expand_ratio[
                    previous : previous + length
                ]
                w[idx] = width[idx]
                parted_subnet = {"d": d, "e": e, "w": w}
                for ele in lookup_table:
                    print(parted_subnet.items())
                    if all(
                        key in ele["config"] and ele["config"][key] == value
                        for key, value in parted_subnet.items()
                    ):
                        params += float(ele["params"])
                        flops += float(ele["flops"])

        return params, flops

    def surrogate_predictor(self, subnet, pretrained_predictor, objs):
        """method to predict performance only for latency or mIoU from given architecture features(subnets)"""
        if "latency" in objs:
            MAX_VALUE = MAX_VALUE_OF_DATASET[0]
            MIN_VALUE = MIN_VALUE_OF_DATASET[0]
        else:
            MAX_VALUE = MAX_VALUE_OF_DATASET[1]
            MIN_VALUE = MIN_VALUE_OF_DATASET[1]
        subnet = self.searchSpace._one_hot_encode(subnet)
        pretrained_list = json.load(open(pretrained_predictor, "r"))
        list = []
        model_list = []
        chunk_size = len(pretrained_list) // 10
        result = 0.0
        for i in range(0, len(pretrained_list), chunk_size):
            list.append(
                dict(itertools.islice(pretrained_list.items(), i, i + chunk_size))
            )

        for i, sublist in enumerate(list):
            # print(f"Sublist {i+1}: {sublist}")
            model_list.append(MosegNASRankNet(pretrained=sublist))

        for model in model_list:
            result_list = model.forward(subnet)
            original_data = (np.array(result_list) - LOWER_BOUND) * (
                MAX_VALUE - MIN_VALUE
            ) / (UPPER_BOUND - LOWER_BOUND) + MIN_VALUE
            original_data = np.exp(original_data)
            result += sum(original_data) / len(original_data)
        return result / len(model_list)

    def predict(self, subnet, true_eval, objs, **kwargs):
        """method to predict performance from given architecture features(subnets)"""
        pred = {}

        if true_eval:
            if "params" or "flops" in objs:
                pred["params"] = self.addup_predictor(subnet=subnet)
            if "latency" in objs:
                pred["latency"] = self.surrogate_predictor(
                    subnet=subnet,
                    pretrained_predictor=self.latency_pretrained,
                    objs="latency",
                )
            if "mIoU" in objs:
                pred["mIoU"] = self.surrogate_predictor(
                    subnet=subnet,
                    pretrained_predictor=self.mIoU_pretrained,
                    objs="mIoU",
                )
            if "err" in objs:
                pred["err"] = 1 - self.surrogate_predictor(
                    subnet=subnet,
                    pretrained_predictor=self.mIoU_pretrained,
                    objs="mIoU",
                )
        else:
            try:
                pred["params"], pred["flops"], pred["latency"], pred["mIoU"] = self.fit(
                    subnet=subnet
                )
            except:
                raise Exception("No result found!")

        return pred
