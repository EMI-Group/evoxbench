import random
import collections
import os
import numpy as np
from pathlib import Path
from copy import deepcopy
from itertools import repeat

from evoxbench.modules import SearchSpace, Evaluator, Benchmark, MLPPredictor


__all__ = ['TransformerSearchSpace', "TransformerEvaluator", "TransformerBenchmark"]


# ------------------- Following functions are specific to Transformer search space ------------------- #
def make_idx_map(entry):
    reverse_map = {}
    for i, v in enumerate(entry):
        reverse_map[v] = i
    return reverse_map

def get_path(name):
    return str(Path(os.environ.get("EVOXBENCH_MODEL", os.getcwd())) / "transformer" / name)


class TransformerFeatureEncoder(object):
    def __init__(self,
                 categories=None
                 ):
        # AutoFormer Base
        self.depth = (14, 15, 16)
        self.embed_dim = (528, 576, 624)
        self.mlp_ratio = (3.0, 3.5, 4.0)
        self.num_heads = (9, 10)

        self.idx_to_depth = make_idx_map(self.depth)
        self.idx_to_embed_dim = make_idx_map(self.embed_dim)
        self.idx_to_mlp_ratio = make_idx_map(self.mlp_ratio)
        self.idx_to_num_heads = make_idx_map(self.num_heads)

        self._max_depth = max(self.depth)

        self.n_var = 2 + 2 * self._max_depth
        # x=[depth, embed_dim, l1_mlp_ratio, l2_mlp_ratio,..., l1_num_heads, l2_num_heads]

        self.lb = [0] + [0] + [0] * self._max_depth + [0] * self._max_depth
        self.ub = [len(self.depth) - 1] + [len(self.embed_dim) - 1] + [len(self.mlp_ratio) - 1] * self._max_depth + [
            len(self.num_heads) - 1] * self._max_depth

        if categories:
            self.categories = categories
        else:
            self.categories = [list(range(a, b + 1)) for a, b in zip(self.lb, self.ub)]

    def arch2x(self, arch):
        # the same as Search Space encode
        depth = arch["depth"]
        embed_dim = arch["embed_dim"]
        mlp_ratio_list = arch["mlp_ratio"]
        num_heads_list = arch["num_heads"]

        if not self.is_valid(depth, embed_dim, mlp_ratio_list, num_heads_list):
            # todo: consider a repair operator or enforce validity in creating an architecture
            raise ValueError("invalid architecture")

        x = np.zeros(self.n_var, dtype=np.int64)
        x[0] = self.idx_to_depth[depth]
        x[1] = self.idx_to_embed_dim[embed_dim]
        x[2:2 + depth] = [self.idx_to_mlp_ratio[mlp_ratio] for mlp_ratio in mlp_ratio_list]
        x[2 + self._max_depth:2 + self._max_depth + depth] = [
            self.idx_to_num_heads[num_heads] for num_heads in num_heads_list]

        return x

    def is_valid(self, depth, embed_dim, mlp_ratio_list, num_heads_list):
        # check validity of arch from outside
        _is_valid = depth in self.depth and embed_dim in self.embed_dim and \
                    len(mlp_ratio_list) == depth and len(num_heads_list) == depth

        for mlp_ratio in mlp_ratio_list:
            if mlp_ratio not in self.mlp_ratio:
                _is_valid = False
                break

        for num_heads in num_heads_list:
            if num_heads not in self.num_heads:
                _is_valid = False
                break

        return _is_valid

    def archs2feature(self, archs):
        # get encoding of arch
        X = [self.arch2x(arch) for arch in archs]
        # Transform using one-hot encoding.
        oh_enc_len = 0
        for cat in self.categories:
            oh_enc_len += len(cat)
        oh_enc = np.zeros((len(X), oh_enc_len))

        for j, x in enumerate(X):
            base = 0
            for i in range(len(x)):
                try:
                    idx = self.categories[i].index(x[i])
                except:
                    raise ValueError("Found unknown categories [{}] in column {}.".format(x[i], i))
                oh_enc[j][base + idx] = 1.0
                base += len(self.categories[i])
        return oh_enc


class TransformerComplexityPredictor:
    def __init__(self):
        self.model = VisionTransformerSuper(embed_dim=640, num_heads=10, depth=16, mlp_ratio=4.0)

    def preporcess(self, sample_config):
        sample_config = deepcopy(sample_config)
        sample_config['embed_dim'] = [sample_config['embed_dim']] * sample_config['depth']
        sample_config['layer_num'] = sample_config['depth']
        return sample_config

    def predict_params(self, arch):
        arch = self.preporcess(arch)
        self.model.set_sample_config(arch)
        params = self.model.get_sampled_params_numel()
        flops = self.model.get_complexity()
        return params

    def predict_flops(self, arch):
        arch = self.preporcess(arch)
        self.model.set_sample_config(arch)
        flops = self.model.get_complexity()
        return flops


def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return tuple(repeat(x, 2))


class Conv2d:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        self.weight = np.zeros([out_channels, in_channels, *kernel_size])
        self.bias = np.zeros(out_channels)


class VisionTransformerSuper:
    def __init__(self, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 img_size=224, patch_size=16, in_chans=3, num_classes=1000, qkv_bias=True, pre_norm=True, gp=True,
                 relative_position=True, change_qkv=True, abs_pos=True, max_relative_position=14):
        # the configs of super arch
        self.super_embed_dim = embed_dim
        # self.super_embed_dim = args.embed_dim
        self.super_mlp_ratio = mlp_ratio
        self.super_layer_num = depth
        self.super_num_heads = num_heads

        self.num_classes = num_classes
        self.pre_norm = pre_norm

        self.patch_embed_super = PatchembedSuper(img_size=img_size, patch_size=patch_size,
                                                 in_chans=in_chans, embed_dim=embed_dim)
        self.gp = gp

        # configs for the sampled subTransformer
        self.sample_embed_dim = None
        self.sample_mlp_ratio = None
        self.sample_layer_num = None
        self.sample_num_heads = None
        self.sample_dropout = None
        self.sample_output_dim = None

        self.blocks = []
        for i in range(depth):
            self.blocks.append(TransformerEncoderLayer(dim=embed_dim,
                                                       num_heads=num_heads,
                                                       mlp_ratio=mlp_ratio,
                                                       qkv_bias=qkv_bias,
                                                       pre_norm=pre_norm,
                                                       change_qkv=change_qkv, relative_position=relative_position,
                                                       max_relative_position=max_relative_position))

        # parameters for vision transformer
        self.num_patches = self.patch_embed_super.num_patches

        self.abs_pos = abs_pos
        if self.abs_pos:
            self.pos_embed = np.zeros((1, self.num_patches + 1, embed_dim))

        self.cls_token = np.zeros((1, 1, embed_dim))

        # self.pos_drop = nn.Dropout(p=drop_rate)
        if self.pre_norm:
            self.norm = LayerNormSuper(super_embed_dim=embed_dim)

        # classifier head
        if num_classes > 0:
            self.head = LinearSuper(embed_dim, num_classes)
        else:
            raise ValueError(
                "num_classes must be > 0 if using a classification head")

    def set_sample_config(self, config: dict):
        self.sample_embed_dim = config['embed_dim']
        self.sample_mlp_ratio = config['mlp_ratio']
        self.sample_layer_num = config['layer_num']
        self.sample_num_heads = config['num_heads']

        self.patch_embed_super.set_sample_config(self.sample_embed_dim[0])
        self.sample_output_dim = [
                                     out_dim for out_dim in self.sample_embed_dim[1:]] + [self.sample_embed_dim[-1]]
        for i, blocks in enumerate(self.blocks):
            # not exceed sample layer number
            if i < self.sample_layer_num:
                blocks.set_sample_config(is_identity_layer=False,
                                         sample_embed_dim=self.sample_embed_dim[i],
                                         sample_mlp_ratio=self.sample_mlp_ratio[i],
                                         sample_num_heads=self.sample_num_heads[i],
                                         sample_out_dim=self.sample_output_dim[i])
            # exceeds sample layer number
            else:
                blocks.set_sample_config(is_identity_layer=True)
        if self.pre_norm:
            self.norm.set_sample_config(self.sample_embed_dim[-1])
        self.head.set_sample_config(
            self.sample_embed_dim[-1], self.num_classes)

    def get_sampled_params_numel(self, config=None):
        if config is not None:
            self.set_sample_config(config)
        numels = []
        numels.append(self.patch_embed_super.calc_sampled_param_num())
        for block in self.blocks[:self.sample_layer_num]:
            for module in [
                block.attn,
                block.attn_layer_norm,
                block.ffn_layer_norm,
                block.fc1,
                block.fc2]:
                numels.append(module.calc_sampled_param_num())

        numels.append(self.norm.calc_sampled_param_num())
        numels.append(self.head.calc_sampled_param_num())
        # print("fake:",numels,len(numels))

        return sum(numels) + self.sample_embed_dim[0] * (2 + self.patch_embed_super.num_patches)

    def get_complexity(self, sequence_length=None):
        if sequence_length is None:
            sequence_length = self.num_patches

        total_flops = 0
        total_flops += self.patch_embed_super.get_complexity(sequence_length)
        total_flops += self.pos_embed[...,
                       :self.sample_embed_dim[0]].size / 2.0
        for blk in self.blocks:
            total_flops += blk.get_complexity(sequence_length + 1)
        total_flops += self.head.get_complexity(sequence_length + 1)
        return total_flops


class TransformerEncoderLayer:
    """Encoder layer block.

    Args:
        args (argparse.Namespace): parsed command-line arguments which
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 pre_norm=True,
                 relative_position=False, change_qkv=False, max_relative_position=14):

        # the configs of super arch of the encoder, three dimension [embed_dim, mlp_ratio, and num_heads]
        self.super_embed_dim = dim
        self.super_mlp_ratio = mlp_ratio
        self.super_ffn_embed_dim_this_layer = int(mlp_ratio * dim)
        self.super_num_heads = num_heads
        self.normalize_before = pre_norm

        self.relative_position = relative_position
        # self.super_activation_dropout = getattr(args, 'activation_dropout', 0)

        # the configs of current sampled arch
        self.sample_embed_dim = None
        self.sample_mlp_ratio = None
        self.sample_ffn_embed_dim_this_layer = None
        self.sample_num_heads_this_layer = None

        self.is_identity_layer = None
        self.attn = AttentionSuper(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, relative_position=self.relative_position,
            change_qkv=change_qkv,
            max_relative_position=max_relative_position
        )

        self.attn_layer_norm = LayerNormSuper(self.super_embed_dim)
        self.ffn_layer_norm = LayerNormSuper(self.super_embed_dim)

        self.fc1 = LinearSuper(super_in_dim=self.super_embed_dim,
                               super_out_dim=self.super_ffn_embed_dim_this_layer)
        self.fc2 = LinearSuper(
            super_in_dim=self.super_ffn_embed_dim_this_layer, super_out_dim=self.super_embed_dim)

    def set_sample_config(self, is_identity_layer, sample_embed_dim=None, sample_mlp_ratio=None, sample_num_heads=None,
                          sample_dropout=None, sample_attn_dropout=None, sample_out_dim=None):

        if is_identity_layer:
            self.is_identity_layer = True
            return

        self.is_identity_layer = False

        self.sample_embed_dim = sample_embed_dim
        self.sample_out_dim = sample_out_dim
        self.sample_mlp_ratio = sample_mlp_ratio
        self.sample_ffn_embed_dim_this_layer = int(
            sample_embed_dim * sample_mlp_ratio)
        self.sample_num_heads_this_layer = sample_num_heads

        self.attn_layer_norm.set_sample_config(
            sample_embed_dim=self.sample_embed_dim)

        self.attn.set_sample_config(sample_q_embed_dim=self.sample_num_heads_this_layer * 64,
                                    sample_num_heads=self.sample_num_heads_this_layer,
                                    sample_in_embed_dim=self.sample_embed_dim)

        self.fc1.set_sample_config(
            sample_in_dim=self.sample_embed_dim, sample_out_dim=self.sample_ffn_embed_dim_this_layer)
        self.fc2.set_sample_config(
            sample_in_dim=self.sample_ffn_embed_dim_this_layer, sample_out_dim=self.sample_out_dim)

        self.ffn_layer_norm.set_sample_config(
            sample_embed_dim=self.sample_embed_dim)

    def get_complexity(self, sequence_length):
        total_flops = 0
        if self.is_identity_layer:
            return total_flops
        total_flops += self.attn_layer_norm.get_complexity(sequence_length + 1)
        total_flops += self.attn.get_complexity(sequence_length + 1)
        total_flops += self.ffn_layer_norm.get_complexity(sequence_length + 1)
        total_flops += self.fc1.get_complexity(sequence_length + 1)
        total_flops += self.fc2.get_complexity(sequence_length + 1)
        return total_flops


class LayerNormSuper:
    def __init__(self, super_embed_dim):
        # the largest embed dim
        self.super_embed_dim = super_embed_dim

        # the current sampled embed dim
        self.sample_embed_dim = None

        self.samples = {}

        self.weight = np.zeros(super_embed_dim)
        self.bias = np.zeros(super_embed_dim)

    def _sample_parameters(self):
        self.samples['weight'] = self.weight[:self.sample_embed_dim]
        self.samples['bias'] = self.bias[:self.sample_embed_dim]
        return self.samples

    def set_sample_config(self, sample_embed_dim):
        self.sample_embed_dim = sample_embed_dim
        self._sample_parameters()

    def calc_sampled_param_num(self):
        assert 'weight' in self.samples.keys()
        assert 'bias' in self.samples.keys()
        return self.samples['weight'].size + self.samples['bias'].size

    def get_complexity(self, sequence_length):
        return sequence_length * self.sample_embed_dim


class AttentionSuper:
    def __init__(self, super_embed_dim, num_heads=8, qkv_bias=False,
                 relative_position=False,
                 num_patches=None, max_relative_position=14, change_qkv=False):
        self.num_heads = num_heads
        head_dim = super_embed_dim // num_heads
        self.super_embed_dim = super_embed_dim

        self.change_qkv = change_qkv
        if change_qkv:
            self.qkv = qkv_super(super_embed_dim, 3 *
                                 super_embed_dim, bias=qkv_bias)
        else:
            self.qkv = LinearSuper(
                super_embed_dim, 3 * super_embed_dim, bias=qkv_bias)

        self.relative_position = relative_position
        if self.relative_position:
            self.rel_pos_embed_k = RelativePosition2D_super(
                super_embed_dim // num_heads, max_relative_position)
            self.rel_pos_embed_v = RelativePosition2D_super(
                super_embed_dim // num_heads, max_relative_position)
        self.max_relative_position = max_relative_position
        self.sample_qk_embed_dim = None
        self.sample_v_embed_dim = None
        self.sample_num_heads = None
        self.sample_in_embed_dim = None

        self.proj = LinearSuper(super_embed_dim, super_embed_dim)

    def set_sample_config(self, sample_q_embed_dim=None, sample_num_heads=None, sample_in_embed_dim=None):

        self.sample_in_embed_dim = sample_in_embed_dim
        self.sample_num_heads = sample_num_heads
        if not self.change_qkv:
            self.sample_qk_embed_dim = self.super_embed_dim
        else:
            self.sample_qk_embed_dim = sample_q_embed_dim

        self.qkv.set_sample_config(
            sample_in_dim=sample_in_embed_dim, sample_out_dim=3 * self.sample_qk_embed_dim)
        self.proj.set_sample_config(
            sample_in_dim=self.sample_qk_embed_dim, sample_out_dim=sample_in_embed_dim)
        if self.relative_position:
            self.rel_pos_embed_k.set_sample_config(
                self.sample_qk_embed_dim // sample_num_heads)
            self.rel_pos_embed_v.set_sample_config(
                self.sample_qk_embed_dim // sample_num_heads)

    def calc_sampled_param_num(self):
        total_num = 0
        total_num += self.qkv.calc_sampled_param_num()
        total_num += self.proj.calc_sampled_param_num()
        if self.relative_position:
            total_num += self.rel_pos_embed_k.calc_sampled_param_num()
            total_num += self.rel_pos_embed_v.calc_sampled_param_num()
        return total_num

    def get_complexity(self, sequence_length):
        total_flops = 0
        total_flops += self.qkv.get_complexity(sequence_length)
        # attn
        total_flops += sequence_length * sequence_length * self.sample_qk_embed_dim
        # x
        total_flops += sequence_length * sequence_length * self.sample_qk_embed_dim
        total_flops += self.proj.get_complexity(sequence_length)
        if self.relative_position:
            total_flops += self.max_relative_position * sequence_length * \
                           sequence_length + sequence_length * sequence_length / 2.0
            total_flops += self.max_relative_position * sequence_length * \
                           sequence_length + sequence_length * self.sample_qk_embed_dim / 2.0
        return total_flops


class RelativePosition2D_super:

    def __init__(self, num_units, max_relative_position):
        super().__init__()

        self.num_units = num_units
        self.max_relative_position = max_relative_position
        # The first element in embeddings_table_v is the vertical embedding for the class
        self.embeddings_table_v = np.zeros(
            (max_relative_position * 2 + 2, num_units))
        self.embeddings_table_h = np.zeros(
            (max_relative_position * 2 + 2, num_units))

        self.sample_head_dim = None
        self.sample_embeddings_table_h = None
        self.sample_embeddings_table_v = None

    def set_sample_config(self, sample_head_dim):
        self.sample_head_dim = sample_head_dim
        self.sample_embeddings_table_h = self.embeddings_table_h[:,
                                         :sample_head_dim]
        self.sample_embeddings_table_v = self.embeddings_table_v[:,
                                         :sample_head_dim]

    def calc_sampled_param_num(self):
        return self.sample_embeddings_table_h.size + self.sample_embeddings_table_v.size


class LinearSuper:
    def __init__(self, super_in_dim, super_out_dim, bias=True):
        # super_in_dim and super_out_dim indicate the largest network!
        self.super_in_dim = super_in_dim
        self.super_out_dim = super_out_dim

        # input_dim and output_dim indicate the current sampled size
        self.sample_in_dim = None
        self.sample_out_dim = None

        self.samples = {}

        self.weight = np.zeros([super_out_dim, super_in_dim])
        if bias:
            self.bias = np.zeros(super_out_dim)
        else:
            self.bias = None

    def set_sample_config(self, sample_in_dim, sample_out_dim):
        self.sample_in_dim = sample_in_dim
        self.sample_out_dim = sample_out_dim

        self._sample_parameters()

    def _sample_parameters(self):
        self.samples['weight'] = LinearSuper.sample_weight(
            self.weight, self.sample_in_dim, self.sample_out_dim)
        self.samples['bias'] = self.bias
        if self.bias is not None:
            self.samples['bias'] = LinearSuper.sample_bias(
                self.bias, self.sample_out_dim)
        return self.samples

    def calc_sampled_param_num(self):
        assert 'weight' in self.samples.keys()
        weight_numel = self.samples['weight'].size

        if self.samples['bias'] is not None:
            bias_numel = self.samples['bias'].size
        else:
            bias_numel = 0

        return weight_numel + bias_numel

    def get_complexity(self, sequence_length):
        total_flops = 0
        total_flops += sequence_length * self.samples['weight'].size
        return total_flops

    @staticmethod
    def sample_weight(weight, sample_in_dim, sample_out_dim):
        sample_weight = weight[:, :sample_in_dim]
        sample_weight = sample_weight[:sample_out_dim, :]

        return sample_weight

    @staticmethod
    def sample_bias(bias, sample_out_dim):
        sample_bias = bias[:sample_out_dim]

        return sample_bias


class qkv_super:
    def __init__(self, super_in_dim, super_out_dim, bias=True):

        # super_in_dim and super_out_dim indicate the largest network!
        self.super_in_dim = super_in_dim
        self.super_out_dim = super_out_dim

        # input_dim and output_dim indicate the current sampled size
        self.sample_in_dim = None
        self.sample_out_dim = None

        self.samples = {}

        self.weight = np.zeros([super_out_dim, super_in_dim])
        if bias:
            self.bias = np.zeros(super_out_dim)
        else:
            self.bias = None

    def set_sample_config(self, sample_in_dim, sample_out_dim):
        self.sample_in_dim = sample_in_dim
        self.sample_out_dim = sample_out_dim

        self._sample_parameters()

    def _sample_parameters(self):
        self.samples['weight'] = qkv_super.sample_weight(
            self.weight, self.sample_in_dim, self.sample_out_dim)
        self.samples['bias'] = self.bias
        if self.bias is not None:
            self.samples['bias'] = qkv_super.sample_bias(
                self.bias, self.sample_out_dim)
        return self.samples

    def calc_sampled_param_num(self):
        assert 'weight' in self.samples.keys()
        weight_numel = self.samples['weight'].size

        if self.samples['bias'] is not None:
            bias_numel = self.samples['bias'].size
        else:
            bias_numel = 0

        return weight_numel + bias_numel

    def get_complexity(self, sequence_length):
        total_flops = 0
        total_flops += sequence_length * self.samples['weight'].size
        return total_flops

    @staticmethod
    def sample_weight(weight, sample_in_dim, sample_out_dim):
        sample_weight = weight[:, :sample_in_dim]
        sample_weight = np.concatenate(
            [sample_weight[i:sample_out_dim:3, :] for i in range(3)], axis=0)

        return sample_weight

    @staticmethod
    def sample_bias(bias, sample_out_dim):
        sample_bias = bias[:sample_out_dim]

        return sample_bias


class PatchembedSuper:
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * \
                      (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = Conv2d(in_chans, embed_dim,
                           kernel_size=patch_size, stride=patch_size)
        self.super_embed_dim = embed_dim

        # sampled_
        self.sample_embed_dim = None
        self.sampled_weight = None
        self.sampled_bias = None

    def set_sample_config(self, sample_embed_dim):
        self.sample_embed_dim = sample_embed_dim
        self.sampled_weight = self.proj.weight[:sample_embed_dim, ...]
        self.sampled_bias = self.proj.bias[:sample_embed_dim, ...]

    def calc_sampled_param_num(self):
        return self.sampled_weight.size + self.sampled_bias.size

    def get_complexity(self, sequence_length):
        total_flops = 0
        if self.sampled_bias is not None:
            total_flops += self.sampled_bias.shape[0]
        total_flops += sequence_length * self.sampled_weight.size
        return total_flops


class TransformerSearchSpace(SearchSpace):
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)

        # AutoFormer Base
        self.depth = (14, 15, 16)
        self.embed_dim = (528, 576, 624)
        self.mlp_ratio = (3.0, 3.5, 4.0)
        self.num_heads = (9, 10)

        self.idx_to_depth = make_idx_map(self.depth)
        self.idx_to_embed_dim = make_idx_map(self.embed_dim)
        self.idx_to_mlp_ratio = make_idx_map(self.mlp_ratio)
        self.idx_to_num_heads = make_idx_map(self.num_heads)

        self._max_depth = max(self.depth)

        self.n_var = 2 + 2 * self._max_depth
        # x=[depth, embed_dim, l1_mlp_ratio, l2_mlp_ratio,..., l1_num_heads, l2_num_heads]

        self.lb = [0] + [0] + [0] * self._max_depth + [0] * self._max_depth
        self.ub = [len(self.depth) - 1] + [len(self.embed_dim) - 1] + [len(self.mlp_ratio) - 1] * self._max_depth + [
            len(self.num_heads) - 1] * self._max_depth

        self.categories = self.categories = [list(range(a, b + 1)) for a, b in zip(self.lb, self.ub)]

    @property
    def name(self):
        return "TransformerSearchSpace"

    def _sample(self, phenotype=True):
        x = np.zeros(self.n_var, dtype=np.int64)

        x[0] = random.choice(self.categories[0])
        x[1] = random.choice(self.categories[1])

        depth = self.depth[x[0]]

        x[2:2 + depth] = [random.choice(options) for options in self.categories[2:2 + depth]]
        x[2 + self._max_depth:2 + self._max_depth + depth] = [
            random.choice(options) for options in self.categories[2 + self._max_depth:2 + self._max_depth + depth]]

        if phenotype:
            return self._decode(x)
        else:
            return x

    def _encode(self, arch):
        depth = arch["depth"]
        embed_dim = arch["embed_dim"]
        mlp_ratio_list = arch["mlp_ratio"]
        num_heads_list = arch["num_heads"]

        if not self.is_valid(depth, embed_dim, mlp_ratio_list, num_heads_list):
            # todo: consider a repair operator or enforce validity in creating an architecture
            raise ValueError("invalid architecture")

        x = np.zeros(self.n_var, dtype=np.int64)
        x[0] = self.idx_to_depth[depth]
        x[1] = self.idx_to_embed_dim[embed_dim]
        x[2:2 + depth] = [self.idx_to_mlp_ratio[mlp_ratio] for mlp_ratio in mlp_ratio_list]
        x[2 + self._max_depth:2 + self._max_depth + depth] = [
            self.idx_to_num_heads[num_heads] for num_heads in num_heads_list]

        return x

    def _decode(self, x):
        assert len(x) == self.n_var, "invalid architecture decision variable x"

        depth = self.depth[x[0]]
        embed_dim = self.embed_dim[x[1]]

        # ignore settings that are outside of maximum depth
        mlp_ratio_list = [self.mlp_ratio[i] for i in x[2:2 + depth]]
        num_heads_list = [self.num_heads[i] for i in x[2 + self._max_depth:2 + self._max_depth + depth]]

        return {
            "depth": depth,
            "embed_dim": embed_dim,
            "mlp_ratio": mlp_ratio_list,
            "num_heads": num_heads_list
        }

    def visualize(self, arch):
        raise NotImplementedError

    def is_valid(self, depth, embed_dim, mlp_ratio_list, num_heads_list):
        # check validity of arch from outside
        _is_valid = depth in self.depth and embed_dim in self.embed_dim and \
                    len(mlp_ratio_list) == depth and len(num_heads_list) == depth

        for mlp_ratio in mlp_ratio_list:
            if mlp_ratio not in self.mlp_ratio:
                _is_valid = False
                break

        for num_heads in num_heads_list:
            if num_heads not in self.num_heads:
                _is_valid = False
                break

        return _is_valid


class TransformerEvaluator(Evaluator):
    def __init__(self,
                 valid_acc_model_path,  # ResNet50 validation acc predictor path
                 test_acc_model_path,  # ResNet50 test acc predictor path
                 objs='err&params&flops',  # objectives to be minimized
                 ):
        super().__init__(objs)
        self.feature_encoder = TransformerFeatureEncoder()
        self.complexity_engine = TransformerComplexityPredictor()

        self.valid_acc_predictor = MLPPredictor(pretrained=valid_acc_model_path)
        self.test_acc_predictor = MLPPredictor(pretrained=test_acc_model_path)

    @property
    def name(self):
        return 'TransformerEvaluator'

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
            stats = {}

            if 'err' in objs:
                stats['err'] = 1 - top1[0]

            if 'params' in objs:
                stats['params'] = self.complexity_engine.predict_params(arch)

            if 'flops' in objs:
                stats['flops'] = self.complexity_engine.predict_flops(arch)

            batch_stats.append(stats)

        return batch_stats


class TransformerBenchmark(Benchmark):
    def __init__(self,
                 valid_acc_model_path=get_path("valid_acc_predictor_checkpoint.json"),  # AutoFormer validation acc predictor path
                 test_acc_model_path=get_path("test_acc_predictor_checkpoint.json"),  # AutoFormer test acc predictor path
                 objs='err&params&flops',  # objectives to be minimized
                 normalized_objectives=False,  # whether to normalize the objectives
                 ):

        search_space = TransformerSearchSpace()
        evaluator = TransformerEvaluator(valid_acc_model_path, test_acc_model_path, objs)

        super().__init__(search_space, evaluator, normalized_objectives)

    @property
    def name(self):
        return 'TransformerBenchmark'

    @property
    def _utopian_point(self):
        """ estimated from sampled architectures, use w/ caution """
        return {
            'err&params': [0.17649999965209962, 43480888],
            'err&flops': [0.17649999965209962, 9223839636],
            'err&params&flops': [0.17649999965209962, 43480888, 9223839636],
        }[self.evaluator.objs]

    @property
    def _nadir_point(self):
        """ estimated from sampled architectures, use w/ caution """
        return {
            'err&params': [0.18321999986877446, 74134144],
            'err&flops': [0.18321999986877446, 15402946152],
            'err&params&flops': [0.18321999986877446, 74134144, 15402946152],
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

    benchmark = TransformerBenchmark(
        "/Users/luzhicha/Dropbox/2022/EvoXBench/python_codes/evoxbench/benchmarks/data/transformer/valid_acc_predictor_checkpoint.json",
        "/Users/luzhicha/Dropbox/2022/EvoXBench/python_codes/evoxbench/benchmarks/data/transformer/test_acc_predictor_checkpoint.json",
        normalized_objectives=False,
    )
    benchmark.debug()