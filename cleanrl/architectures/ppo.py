import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from .common import Predictor, layer_init, NormalizeImg


class PPODefault(Predictor):
    def __init__(self, envs, device, normalize=False):
        super().__init__()
        self.device = device

        dims = envs.observation_space.shape

        self.features = nn.Sequential(
            layer_init(nn.Conv2d(dims[0], 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU()
        )

        if normalize:  # x / 255
            self.features.insert(0, NormalizeImg())

        self.flatten = nn.Flatten()
        
        # compute flatten size with a dummy forward
        # makes the agent applicable for any input image size
        with torch.no_grad():
            f = self.features(torch.zeros(dims))
            feat_dim = np.prod(f.shape)

        self.mlp = nn.Sequential(
            layer_init(nn.Linear(feat_dim, 512)),
            nn.ReLU(),
        )
        
        self.actor = layer_init(nn.Linear(512, envs.action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def _embed(self, x):
        return self.mlp(self.flatten(self.features(x.float())))
    
    def get_value(self, x):
        return self.critic(self._embed(x))

    def get_action_and_value(self, x, action=None):
        z = self._embed(x)
        logits = self.actor(z)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(z)


class PPObj(Predictor):
    def __init__(self, envs, device, encoder_dims=(128, 64), decoder_dims=(32,)):
        super().__init__()
        self.device = device

        dims = envs.observation_space.shape
        layers = nn.ModuleList()

        in_dim = dims[-1]

        for l in encoder_dims:
            layers.append(layer_init(nn.Linear(in_dim, l)))
            layers.append(nn.ReLU())
            in_dim = l
        layers.append(nn.Flatten())
        in_dim *= np.prod(dims[:-1], dtype=int)
        l = in_dim
        for l in decoder_dims:
            layers.append(layer_init(nn.Linear(in_dim, l)))
            layers.append(nn.ReLU())
            in_dim = l

        self.network = nn.Sequential(*layers)
        self.actor = layer_init(nn.Linear(l, envs.action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(l, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


class PPO_Obj(Predictor):
    def __init__(self, envs, device):
        super().__init__()
        self.device = device

        self.network = nn.Sequential(
            layer_init(nn.Linear(4,32)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, envs.action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)
