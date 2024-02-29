import os
import random
import json
import math
from pathlib import Path
import numpy as np
import itertools
import time

from evoxbench.modules import SearchSpace, Evaluator, Benchmark, SurrogateModel

__all__ = [
    "MoSegNASSearchSpace",
    "MoSegNASEvaluator",
    "MoSegNASBenchmark",
    "MoSegNASSurrogateModel",
]


def get_path(name):
    return str(Path(os.environ.get("EVOXBENCH_MODEL", os.getcwd())) / "moseg" / name)


class MoSegNASSearchSpace(SearchSpace):
    def __init__(self, subnet_str=True, **kwargs):
        super().__init__(**kwargs)
        self.subnet_str = subnet_str
        self.depth_list = [2, 2, 2, 3, 2]
        self.expand_ratio_list = [0.2, 0.25, 0.35]
        self.width_list = [2, 2, 2, 2, 2, 2]
        self.categories = [list(range(d + 1)) for d in self.depth_list]
        self.categories += [list(range(3))] * 13
        self.categories += [list(range(3))] * 6
        self.n_var = len(self.categories)
        self.lb = [0] * self.n_var
        self.ub = [len(cat) - 1 for cat in self.categories]
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
        e = [self.expand_ratio_list[i] for i in x[5:-6]]
        return {"d": x[:5].tolist(), "e": e, "w": x[-6:].tolist()}

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
        surrogate_pretrained_list=get_path("ranknet_mIoU.json"),
        lookup_table=get_path(
            "ofa_fanet_plus_bottleneck_rtx_params_flops_lookup_ALL.json"
        ),
        objs="err&h1_latency&h2_latency&h1_energy_consumption&h2_energy_consumption&flops&params",
        **kwargs
    ):
        self.search_space = MoSegNASSearchSpace()
        self.surrogate_pretrained_list = surrogate_pretrained_list
        self.lookup_table = lookup_table
        self.objs = objs
        self.evaluator = MoSegNASEvaluator(
            objs=self.objs,
            surrogate_pretrained_list=surrogate_pretrained_list,
            lookup_table=lookup_table,
        )
        super().__init__(self.search_space, self.evaluator, normalized_objectives)

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
        """estimated from sampled architectures, use w/ caution"""
        return {
            "err&h1_latency": np.array([0.0000, 1.9741]),
            "err&h1_latency&flops": np.array([0.0000, 1.9741, 331067392]),
            "err&h1_latency&params": np.array([0.0000, 1.9741, 132512]),
            "err&h1_latency&h1_energy_consumption&flops": np.array(
                [0.0000, 1.9741, 678.0691752185481, 331067392]
            ),
            "err&h1_latency&h1_energy_consumption&flops&params": np.array(
                [0.0000, 1.9741, 678.0691752185481, 331067392, 132512]
            ),
            "err&h2_latency": np.array([0.0000, 58.74647839317504]),
            "err&h2_latency&flops": np.array([0.0000, 58.74647839317504, 331067392]),
            "err&h2_latency&params": np.array([0.0000, 58.74647839317504, 132512]),
            "err&h2_latency&h2_energy_consumption&flops": np.array(
                [0.0000, 58.74647839317504, 734.333948345669, 331067392]
            ),
            "err&h2_latency&h2_energy_consumption&flops&params": np.array(
                [0.0000, 58.74647839317504, 734.333948345669, 331067392, 132512]
            ),
            "err&h1_latency&h2_latency": np.array([0.0000, 1.9741, 58.74647839317504]),
            "err&h1_latency&h2_latency&h1_energy_consumption&h2_energy_consumption": np.array(
                [0.0000, 1.9741, 58.74647839317504, 678.0691752185481, 734.333948345669]
            ),
            "err&h1_latency&h2_latency&h1_energy_consumption&h2_energy_consumption&flops": np.array(
                [
                    0.0000,
                    1.9741,
                    58.74647839317504,
                    678.0691752185481,
                    734.333948345669,
                    331067392,
                ]
            ),
            "err&h1_latency&h2_latency&h1_energy_consumption&h2_energy_consumption&params": np.array(
                [
                    0.0000,
                    1.9741,
                    58.74647839317504,
                    678.0691752185481,
                    734.333948345669,
                    132512,
                ]
            ),
            "err&h1_latency&h2_latency&h1_energy_consumption&h2_energy_consumption&flops&params": np.array(
                [
                    0.0000,
                    1.9741,
                    58.74647839317504,
                    678.0691752185481,
                    734.333948345669,
                    331067392,
                    132512,
                ]
            ),
        }[self.evaluator.objs]

    @property
    def _nadir_point(self):
        """estimated from sampled architectures, use w/ caution"""
        return {
            "err&h1_latency": np.array([1.0000, 11.0309]),
            "err&h1_latency&flops": np.array([1.0000, 11.0309, 1274736640]),
            "err&h1_latency&params": np.array([1.0000, 11.0309, 453224]),
            "err&h1_latency&h1_energy_consumption&flops": np.array(
                [1.0000, 11.0309, 5019.1308754592665, 1274736640]
            ),
            "err&h1_latency&h1_energy_consumption&flops&params": np.array(
                [1.0000, 11.0309, 5019.1308754592665, 1274736640, 453224]
            ),
            "err&h2_latency": np.array([1.0000, 237.3906080799434]),
            "err&h2_latency&flops": np.array([1.0000, 237.3906080799434, 1274736640]),
            "err&h2_latency&params": np.array([1.0000, 237.3906080799434, 453224]),
            "err&h2_latency&h2_energy_consumption&flops": np.array(
                [1.0000, 237.3906080799434, 2967.383299453641, 1274736640]
            ),
            "err&h2_latency&h2_energy_consumption&flops&params": np.array(
                [1.0000, 237.3906080799434, 2967.383299453641, 1274736640, 453224]
            ),
            "err&h1_latency&h2_latency": np.array([1.0000, 11.0309, 237.3906080799434]),
            "err&h1_latency&h2_latency&h1_energy_consumption&h2_energy_consumption": np.array(
                [
                    1.0000,
                    11.0309,
                    237.3906080799434,
                    5019.1308754592665,
                    2967.383299453641,
                ]
            ),
            "err&h1_latency&h2_latency&h1_energy_consumption&h2_energy_consumption&flops": np.array(
                [
                    1.0000,
                    11.0309,
                    237.3906080799434,
                    5019.1308754592665,
                    2967.383299453641,
                    1274736640,
                ]
            ),
            "err&h1_latency&h2_latency&h1_energy_consumption&h2_energy_consumption&params": np.array(
                [
                    1.0000,
                    11.0309,
                    237.3906080799434,
                    5019.1308754592665,
                    2967.383299453641,
                    453224,
                ]
            ),
            "err&h1_latency&h2_latency&h1_energy_consumption&h2_energy_consumption&flops&params": np.array(
                [
                    1.0000,
                    11.0309,
                    237.3906080799434,
                    5019.1308754592665,
                    2967.383299453641,
                    1274736640,
                    453224,
                ]
            ),
        }[self.evaluator.objs]


class MoSegNASEvaluator(Evaluator):
    def __init__(self, objs, surrogate_pretrained_list, lookup_table):
        super().__init__(objs)
        self.surrogate_pretrained_list = surrogate_pretrained_list
        self.feature_encoder = MoSegNASSearchSpace()
        self.surrogate_model = MoSegNASSurrogateModel(
            surrogate_pretrained_list=self.surrogate_pretrained_list,
            categories=self.feature_encoder.categories,
            lookup_table=lookup_table,
        )

    @property
    def name(self):
        return "MoSegNASEvaluator"

    def evaluate(
        self,
        archs,
        objs=None,
        true_eval=True,
    ):
        """evalute the performance of the given subnets"""
        batch_stats = []

        for index, subnet_encoded in enumerate(archs):
            pred = self.surrogate_model.predict(
                subnet=subnet_encoded, true_eval=true_eval, objs=self.objs
            )

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
        return 0.0 if random.random() < self.drop else x / (1 - self.drop)

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
        for layer in range(1, self.n_hidden_layers + 1):
            outputs = self.linear(outputs, self.weights[layer], self.biases[layer])
            outputs = [self.relu(x) for x in outputs]

        # Dropout layer
        outputs = [self.dropout(x) for x in outputs]

        # Output layer
        outputs = self.linear(outputs, self.weights[-1], self.biases[-1])

        # Output -> [0, 1]
        # outputs = [self.sigmoid(x) for x in outputs]
        return outputs

    def train(self):
        pass


class MoSegNASSurrogateModel(SurrogateModel):
    def __init__(
        self,
        surrogate_pretrained_list,
        lookup_table,
        pretrained_result=None,
        categories=[list(range(4))] * 24,
        **kwargs
    ):
        super().__init__()
        self.pretrained_result = pretrained_result
        self.surrogate_pretrained_list = surrogate_pretrained_list
        self.lookup_table = json.load(open(lookup_table, "r"))
        self.head_lookup_table = []
        index = -1
        for i in range(len(self.lookup_table)):
            if self.lookup_table[i]["type"] == "HEAD":
                self.head_lookup_table.append(self.lookup_table[i])
            else:
                index = i
                break
        self.lookup_table = self.lookup_table[index:]
        self.categories = categories
        self.searchSpace = MoSegNASSearchSpace()

        pretrained_list = open(surrogate_pretrained_list, "r")
        pretrained_list = json.load(pretrained_list)
        list = []
        model_list = []
        chunk_size = len(pretrained_list) // 10
        for i in range(0, len(pretrained_list), chunk_size):
            list.append(
                dict(itertools.islice(pretrained_list.items(), i, i + chunk_size))
            )

        for i, sublist in enumerate(list):
            model_list.append(MosegNASRankNet(pretrained=sublist))

        self.mIoU_pretrained = model_list

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
                    return [result["params"], result["flops"], 1 - result["mIoU"]]
        return None

    def head_predictor(self, depth_head, width_head):
        for ele in self.head_lookup_table:
            if (
                depth_head == ele["config"]["depth"]
                and width_head == ele["config"]["width_mult"]
            ):
                return (
                    float(ele["batch_params"]),
                    float(ele["batch_flops"]),
                    float(ele["h1_latency"]),
                    float(ele["h2_latency"]),
                    float(ele["h1_consumption"]),
                    float(ele["h2_consumption"]),
                )

    def if_head_only(self, subnet):
        for i in subnet[1:5]:
            if i != 0:
                return False
        return True

    def addup_predictor(self, subnet):
        """method to predict performance only for flops or params from given architecture features(subnets)"""
        if self.if_head_only(subnet):
            return (
                453224,
                1274736640,
                11.0309,
                237.3906080799434,
                5019.1308754592665,
                2967.383299453641,
            )
        else:
            d_len = len(self.searchSpace.depth_list)
            w_len = len(self.searchSpace.width_list)
            e_len = len(self.searchSpace.categories) - d_len - w_len

            depth = subnet[:d_len]
            expand_ratio = subnet[d_len : d_len + e_len]
            width = subnet[e_len + d_len :]

            mult_list = [0.2, 0.25, 0.35]
            width_head = [mult_list[i] for i in width[0:2]]
            width_head.sort()
            (
                params,
                flops,
                h1_latency,
                h2_latency,
                h1_energy_consumption,
                h2_energy_consumption,
            ) = self.head_predictor([depth[0]], width_head)

            (
                base_params,
                base_flops,
                base_h1_latency,
                base_h2_latency,
                base_h1_energy_consumption,
                base_h2_energy_consumption,
            ) = self.head_predictor([0], [0.2, 0.2])

            cell1 = {
                "d": [0, depth[1]],
                "e": expand_ratio[0:3],
                "w": [0.2, 0.2, mult_list[width[2]]],
            }
            cell2 = {
                "d": [0, depth[2]],
                "e": expand_ratio[3:6],
                "w": [0.2, 0.2, mult_list[width[3]]],
            }
            cell3 = {
                "d": [0, depth[3]],
                "e": expand_ratio[6:10],
                "w": [0.2, 0.2, mult_list[width[4]]],
            }
            cell4 = {
                "d": [0, depth[4]],
                "e": expand_ratio[10:13],
                "w": [0.2, 0.2, mult_list[width[5]]],
            }
            for cell in [cell1, cell2, cell3, cell4]:
                if cell["d"][1] != 0:
                    cell["e"].sort()
                    cell["d"].sort()
                    cell["w"].sort()
                    for ele in self.lookup_table:
                        if all(
                            key in ele and ele[key] == value
                            for key, value in cell.items()
                        ):
                            params += float(ele["batch_params"]) - base_params
                            flops += float(ele["batch_flops"]) - base_flops
                            h1_latency += float(ele["h1_latency"]) - base_h1_latency
                            h2_latency += float(ele["h2_latency"]) - base_h2_latency
                            h1_energy_consumption += (
                                float(ele["h1_consumption"])
                                - base_h1_energy_consumption
                            )
                            h2_energy_consumption += (
                                float(ele["h2_consumption"])
                                - base_h2_energy_consumption
                            )
                            break
        h1_latency = h1_latency * random.uniform(0.98, 1.02)
        h2_latency = h2_latency * random.uniform(0.98, 1.02)
        h1_energy_consumption = h1_energy_consumption * random.uniform(0.95, 1.05)
        h2_energy_consumption = h2_energy_consumption * random.uniform(0.95, 1.05)
        return (
            params,
            flops,
            h1_latency,
            h2_latency,
            h1_energy_consumption,
            h2_energy_consumption,
        )

    def surrogate_predictor(self, subnet, pretrained_predictor, true_eval):
        """method to predict performance only for latency or err from given architecture features(subnets)"""
        if true_eval:
            pretrained_predictor = [pretrained_predictor]
            result = 0
            return_list = []
            for model in pretrained_predictor:
                result_list = model.forward(subnet)
                return_list.append(result_list[0])
            result = sum(result_list) / len(result_list)
            return result
        else:
            return pretrained_predictor.forward(subnet)[0]

    def predict(self, subnet, true_eval, objs, **kwargs):
        """method to predict performance from given architecture features(subnets)"""
        pred = {}
        if true_eval:
            if (
                "params"
                or "flops"
                or "h1_latency"
                or "h2_latency"
                or "h1_energy_consumption"
                or "h2_energy_consumption" in objs
            ):
                subnet2 = subnet["d"] + subnet["e"] + subnet["w"]
                (
                    pred["params"],
                    pred["flops"],
                    pred["h1_latency"],
                    pred["h2_latency"],
                    pred["h1_energy_consumption"],
                    pred["h2_energy_consumption"],
                ) = self.addup_predictor(subnet=subnet2)
            if "err" in objs:
                subnet = self.searchSpace._encode(subnet)
                subnet = self.searchSpace._one_hot_encode(subnet)
                pred["err"] = 1 - self.surrogate_predictor(
                    subnet=subnet,
                    pretrained_predictor=self.mIoU_pretrained[0],
                    true_eval=true_eval,
                )
                pred["err"] = min(pred["err"], 1.0000)
                pred["err"] = max(pred["err"], 0.0000)
        else:
            if (
                "params"
                or "flops"
                or "h1_latency"
                or "h2_latency"
                or "h1_energy_consumption"
                or "h2_energy_consumption" in objs
            ):
                subnet2 = subnet["d"] + subnet["e"] + subnet["w"]
                (
                    pred["params"],
                    pred["flops"],
                    pred["h1_latency"],
                    pred["h2_latency"],
                    pred["h1_energy_consumption"],
                    pred["h2_energy_consumption"],
                ) = self.addup_predictor(subnet=subnet2)
            if "err" in objs:
                subnet = self.searchSpace._encode(subnet)
                subnet = self.searchSpace._one_hot_encode(subnet)
                pred["err"] = 1 - self.surrogate_predictor(
                    subnet=subnet,
                    pretrained_predictor=self.mIoU_pretrained[0],
                    true_eval=true_eval,
                )
                pred["err"] = min(pred["err"], 1.0000)
                pred["err"] = max(pred["err"], 0.0000)

        result = {}
        for _ in objs.split("&"):
            result[_] = pred[_]
        return result
