import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions.categorical import Categorical

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PPODefault(nn.Module):
    
    def __init__(self, envs, device):
        super().__init__()
        self.device = device
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
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

    def predict(self, x, states=None, **_):
        with torch.no_grad():
            return np.argmax(self.actor(self.network(torch.Tensor(x).to(self.device))).cpu().numpy(), axis=1), states

class PPO_Obj_small(nn.Module):
    def __init__(self, envs, device):
        super().__init__()
        self.device = device

        self.network = nn.Sequential(
            layer_init(nn.Linear(4,128)),
            nn.ReLU(),
            layer_init(nn.Linear(128,64)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64*4*3, 32)),
            nn.ReLU(),
            
        )
        self.actor = layer_init(nn.Linear(32, envs.action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(32, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

    def predict(self, x, states=None, **_):
        with torch.no_grad():
            return np.argmax(self.actor(self.network(torch.Tensor(x).to(self.device))).cpu().numpy(), axis=1), states

class PPO_Obj(nn.Module):
    def __init__(self, envs, device):
        super().__init__()
        self.device = device

        self.network = nn.Sequential(
            layer_init(nn.Linear(4,32)),
            nn.ReLU(),
            layer_init(nn.Linear(32,64)),
            nn.ReLU(),
            layer_init(nn.Linear(64,64)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64*4*3, 512)),
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

    def predict(self, x, states=None, **_):
        with torch.no_grad():
            return np.argmax(self.actor(self.network(torch.Tensor(x).to(self.device))).cpu().numpy(), axis=1), states