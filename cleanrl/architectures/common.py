import torch
import torch.nn as nn

import numpy as np


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Predictor(nn.Module):
    def predict(self, x, states=None, **_):
        with torch.no_grad():
            return np.argmax(self.actor(self.network(torch.Tensor(x).to(self.device))).cpu().numpy(), axis=1), states