import pickle

"""
This is the class for the API for HW-NAS-Bench
"""


class HWNASBenchAPI():
    def __init__(self, file_path_or_dict, search_space="nasbench201"):
        # Load the pickle file containing all the Hardware metrics
        with open(file_path_or_dict, 'rb') as f:
            self.HW_metrics = pickle.load(f)
        self.search_space = search_space

    def query_by_index(self, arch_index, dataname):
        if self.search_space == "nasbench201":
            metrics_list = ["edgegpu_latency",
                            "edgegpu_energy",
                            "raspi4_latency",
                            "edgetpu_latency",
                            "pixel3_latency",
                            "eyeriss_latency",
                            "eyeriss_energy",
                            "eyeriss_arithmetic_intensity",
                            "fpga_latency",
                            "fpga_energy",
                            ]
            results = {}
            for metric_name in metrics_list:
                results[metric_name] = float(self.HW_metrics[self.search_space][dataname][metric_name][arch_index])

            # for 'average_hw_metric'
            results['average_hw_metric'] = results["edgegpu_latency"] * results["edgegpu_energy"] * results[
                "raspi4_latency"] * results["edgetpu_latency"] * results["pixel3_latency"] * results[
                                               "eyeriss_latency"] * results["eyeriss_energy"] * results[
                                               "fpga_latency"] * results["fpga_energy"]

            return results
        elif self.search_space == "fbnet":
            metrics_list = ["edgegpu_latency",
                            "edgegpu_energy",
                            "raspi4_latency",
                            "pixel3_latency",
                            "eyeriss_latency",
                            "eyeriss_energy",
                            "fpga_latency",
                            "fpga_energy",
                            ]
            results = {}
            for metric_name in metrics_list:
                lookup_table = self.HW_metrics[self.search_space][metric_name]
                results[metric_name] = fbnet_get_metrics(arch_index, dataname, lookup_table)

            # for 'average_hw_metric'
            results['average_hw_metric'] = results["edgegpu_latency"] * results["edgegpu_energy"] * results[
                "raspi4_latency"] * results["pixel3_latency"] * results["eyeriss_latency"] * results["eyeriss_energy"] * \
                                           results["fpga_latency"] * results["fpga_energy"]

            return results
        else:
            print("Wrong dataset name, expect: {} and {}, while receive: {}",
                  format("nasbench201", "fbnet", self.search_space))

    def get_net_config(self, arch_index, dataname):
        if self.search_space == "nasbench201":
            config = self.HW_metrics[self.search_space][dataname]["config"][arch_index]
            return config
        elif self.search_space == "fbnet":
            ops_str_lookup_table = ["k3_e1", "k3_e1_g2", "k3_e3",
                                    "k3_e6", "k5_e1", "k5_e1_g2",
                                    "k5_e3", "k5_e6", "skip"]
            assert dataname in ["cifar100", "ImageNet"], (
                "Only {} and {} are allowed to be datasets in FBNet space, while receive {}".format("cifar100",
                                                                                                    "ImageNet",
                                                                                                    dataname))
            if dataname == "cifar100":
                num_classes = 100
            else:
                num_classes = 1000  # ImageNet
            arch_str = [ops_str_lookup_table[v] for v in arch_index]
            config = {"dataset": dataname, "num_classes": num_classes, "op_idx_list": arch_index, "arch_str": arch_str}
            return config
        else:
            print("Wrong dataset name, expect: {} and {}, while receive: {}",
                  format("nasbench201", "fbnet", self.search_space))

    def get_op_lookup_tables(self):
        if self.search_space == "fbnet":
            lookup_tables = {}
            metrics_list = ["edgegpu_latency",
                            "edgegpu_energy",
                            "raspi4_latency",
                            "pixel3_latency",
                            "eyeriss_latency",
                            "eyeriss_energy",
                            "fpga_latency",
                            "fpga_energy",
                            ]
            results = {}
            for metric_name in metrics_list:
                lookup_table = self.HW_metrics[self.search_space][metric_name]
                lookup_tables[metric_name] = lookup_table
            return lookup_tables
        else:
            print("Wrong dataset name, expect: {} and {}, while receive: {}",
                  format("nasbench201", "fbnet", self.search_space))


def fbnet_get_metrics(arch_index, dataname, lookup_table):
    # Receive arch_index (a list format), dataname ("cifar100", "ImageNet"), and an lookup_table
    assert type(arch_index) == list, (
        "The type of arch_index should be a list in FBNet space, while receive {}".format(type(arch_index)))
    assert len(arch_index) == 22, (
        "The length of arch_index should be 22 in FBNet space, while receive {}".format(len(arch_index)))
    assert dataname in ["cifar100", "ImageNet"], (
        "Only {} and {} are allowed to be datasets in FBNet space, while receive {}".format("cifar100", "ImageNet",
                                                                                            dataname))

    # Architecture settings in FBNet space
    op_idx_list = arch_index
    OP_metrics_dict = lookup_table

    stem_channel = 16
    num_layer_list = [1, 4, 4, 4, 4, 4, 1]
    num_channel_list = [16, 24, 32, 64, 112, 184, 352]

    ConvBlock_list = []
    ConvBlock_list.append({"exp": 1, "kernel": 3, "group": 1})
    ConvBlock_list.append({"exp": 1, "kernel": 3, "group": 2})
    ConvBlock_list.append({"exp": 3, "kernel": 3, "group": 1})
    ConvBlock_list.append({"exp": 6, "kernel": 3, "group": 1})
    ConvBlock_list.append({"exp": 1, "kernel": 5, "group": 1})
    ConvBlock_list.append({"exp": 1, "kernel": 5, "group": 2})
    ConvBlock_list.append({"exp": 3, "kernel": 5, "group": 1})
    ConvBlock_list.append({"exp": 6, "kernel": 5, "group": 1})

    if dataname == "cifar100":
        num_classes = 100
        stride_list = [1, 1, 2, 2, 1, 2, 1]
        header_channel = 1504  # FBNetv1 setting
        stride_init = 1
        H_W = 32
    else:
        num_classes = 1000  # ImageNet
        stride_list = [1, 2, 2, 2, 1, 2, 1]  # FBNetv1 setting, offcial ImageNet setting
        header_channel = 1984  # FBNetv1 setting
        stride_init = 2
        H_W = 224

    metric = 0.0

    # Add STEM cost
    metric += OP_metrics_dict["ConvNorm_H{}_W{}_Cin{}_Cout{}_kernel{}_stride{}_group{}".format(
        H_W,
        H_W,
        3,
        stem_channel,
        3,
        stride_init,
        1,
    )]
    H_W = H_W // stride_init  # Downsample size due to stride
    # Add Cells cost
    layer_id = 0
    for stage_id, num_layer in enumerate(num_layer_list):
        for i in range(num_layer):
            layer_op_idx = op_idx_list[layer_id]
            if i == 0:  # first Conv takes the stride into consideration
                if stage_id == 0:
                    # The first block in the first stage will use stem_channel as input channel
                    if layer_op_idx < 8:  # ConvBlock
                        metric += OP_metrics_dict[
                            "ConvBlock_H{}_W{}_Cin{}_Cout{}_exp{}_kernel{}_stride{}_group{}".format(
                                H_W,
                                H_W,
                                stem_channel,
                                num_channel_list[stage_id],
                                ConvBlock_list[layer_op_idx]["exp"],
                                ConvBlock_list[layer_op_idx]["kernel"],
                                stride_list[stage_id],
                                ConvBlock_list[layer_op_idx]["group"],
                            )]
                    else:  # Skip connection
                        metric += OP_metrics_dict["Skip_H{}_W{}_Cin{}_Cout{}_stride{}".format(
                            H_W,
                            H_W,
                            stem_channel,
                            num_channel_list[stage_id],
                            stride_list[stage_id]
                        )]
                else:
                    if layer_op_idx < 8:  # ConvBlock
                        metric += OP_metrics_dict[
                            "ConvBlock_H{}_W{}_Cin{}_Cout{}_exp{}_kernel{}_stride{}_group{}".format(
                                H_W,
                                H_W,
                                num_channel_list[stage_id - 1],
                                num_channel_list[stage_id],
                                ConvBlock_list[layer_op_idx]["exp"],
                                ConvBlock_list[layer_op_idx]["kernel"],
                                stride_list[stage_id],
                                ConvBlock_list[layer_op_idx]["group"],
                            )]
                    else:  # Skip connection
                        metric += OP_metrics_dict["Skip_H{}_W{}_Cin{}_Cout{}_stride{}".format(
                            H_W,
                            H_W,
                            num_channel_list[stage_id - 1],
                            num_channel_list[stage_id],
                            stride_list[stage_id]
                        )]
                H_W = H_W // stride_list[stage_id]  # Downsample size due to stride
            else:
                if layer_op_idx < 8:  # ConvBlock
                    metric += OP_metrics_dict["ConvBlock_H{}_W{}_Cin{}_Cout{}_exp{}_kernel{}_stride{}_group{}".format(
                        H_W,
                        H_W,
                        num_channel_list[stage_id],
                        num_channel_list[stage_id],
                        ConvBlock_list[layer_op_idx]["exp"],
                        ConvBlock_list[layer_op_idx]["kernel"],
                        1,
                        ConvBlock_list[layer_op_idx]["group"],
                    )]
                else:  # Skip connection
                    metric += OP_metrics_dict["Skip_H{}_W{}_Cin{}_Cout{}_stride{}".format(
                        H_W,
                        H_W,
                        num_channel_list[stage_id],
                        num_channel_list[stage_id],
                        1
                    )]
                # no downsample size because of stride = 1
            layer_id += 1
    # Add Header cost
    metric += OP_metrics_dict["ConvNorm_H{}_W{}_Cin{}_Cout{}_kernel{}_stride{}_group{}".format(
        H_W,
        H_W,
        num_channel_list[-1],
        header_channel,
        1,
        1,
        1,
    )]
    # Add AvgP cost
    metric += OP_metrics_dict["AvgP_H{}_W{}_Cin{}_Cout{}_kernel{}_stride{}".format(
        H_W,
        H_W,
        header_channel,
        header_channel,
        H_W,
        1,
    )]
    # Add FC cost
    metric += OP_metrics_dict["FC_Cin{}_Cout{}".format(
        header_channel,
        num_classes,
    )]

    return metric
