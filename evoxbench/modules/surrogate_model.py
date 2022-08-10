import json
import numpy as np
from abc import ABC, abstractmethod

__all__ = ['SurrogateModel', 'MLPPredictor']


class SurrogateModel(ABC):
    def __init__(self, **kwargs):
        pass

    @property
    @abstractmethod
    def name(self):
        """ name of the surrogate model """
        raise NotImplementedError

    @abstractmethod
    def fit(self, X, **kwargs):
        """ method to fit/learn/train a surrogate model from data """
        raise NotImplementedError

    @abstractmethod
    def predict(self, features, **kwargs):
        """ method to predict performance from architecture features """
        raise NotImplementedError


class MLPPredictor(SurrogateModel):
    def __init__(self,
                 pretrained,  # pretrained weights and bias dict
                 ):

        super().__init__()

        # weights = json.load(open(pretrained, 'r'))
        checkpoints = json.load(open(pretrained, 'r'))

        weights = checkpoints['state_dicts']
        self.remap_factor = checkpoints['remap_factor']
        self.base_acc = checkpoints['base_acc']

        if not isinstance(weights, list):
            weights = [weights]

        models = []
        for weight in weights:
            model = {}
            for key, values in weight.items():
                model[key] = np.array(values)

            model['normalization_factor'] = weight['normalization_factor']
            models.append(model)

        self.models = models

        # self.model = {}
        # for key, values in json.load(open(pretrained, 'r')).items():
        #     self.model[key] = np.array(values)

    @property
    def name(self):
        return 'MLP Predictor'

    def fit(self, X, **kwargs):
        raise NotImplementedError

    def forward(self, X, model):
        for i in range(1, 4):
            X = np.matmul(X, model['W{}'.format(i)].transpose()) + model['b{}'.format(i)][None, :]
            X = np.maximum(X, 0)

        X = np.matmul(X, model['W4'].transpose())
        X += model['b4'][None, :]
        # if 'b4' in model:
        #

        # if 'base_acc' in model:
        #     X += model['base_acc']

        # if 'normalization_factor' in model:
        #     X = (X - model['normalization_factor'][0]) / model['normalization_factor'][1]

        # if 'remap_factor' in model:
        #     X = X * model['remap_factor'][1] + model['remap_factor'][0]

        # return X
        X += self.base_acc
        # normalization
        X = (X - model['normalization_factor'][0]) / model['normalization_factor'][1]

        return X * self.remap_factor[1] + self.remap_factor[0]  # remap to range

    def predict(self, features, is_noisy=False, **kwargs):

        if is_noisy:
            pred = self.forward(features, np.random.choice(self.models))
        else:
            pred = 0
            for model in self.models:
                pred += self.forward(features, model)

            pred = pred / len(self.models)
        return pred


if __name__ == '__main__':
    import copy
    import torch

    state_dict = torch.load("/Users/luzhicha/Dropbox/2022/EvoXBench/python_codes/data/mnv3/acc_predictor.pth",
                            map_location=torch.device("cpu"))

    for key, values in state_dict.items():
        print(key)
        print(values.size())
    exit()
    new_state_dict = {}
    w_i, b_i = 1, 1
    for key, values in state_dict.items():
        if 'weight' in key:
            new_state_dict['W{}'.format(w_i)] = copy.deepcopy(values.cpu().detach().numpy().tolist())
            w_i += 1
        if 'bias' in key:
            new_state_dict['b{}'.format(b_i)] = copy.deepcopy(values.cpu().detach().numpy().tolist())
            b_i += 1

    for key, values in new_state_dict.items():
        print(key)
        print(values)

    with open('/Users/luzhicha/Dropbox/2022/EvoXBench/python_codes/data/mnv3/acc_predictor.json', 'w') as fp:
        json.dump(new_state_dict, fp)
