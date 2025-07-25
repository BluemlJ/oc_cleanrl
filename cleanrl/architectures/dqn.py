import torch
import torch.nn as nn

from .common import Predictor, layer_init


# ALGO LOGIC: initialize agent here:
class QNetwork(Predictor):
    def __init__(self, env):
        super().__init__()

        dims = env.observation_space.shape

        self.network = nn.Sequential(
            nn.Conv2d(dims[0], 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, env.action_space.n),
        )

    def forward(self, x):
        return self.network(x / 255.0)

    def get_action_and_value(self, x):
        q_values = self.network(x / 255.0)
        action = torch.argmax(q_values, 1)
        return action, q_values[:, action], None, None


class QNetwork_C51(Predictor):
    def __init__(self, env, n_atoms=101, v_min=-100, v_max=100):
        super().__init__()
        self.env = env
        self.n_atoms = n_atoms
        self.register_buffer("atoms", torch.linspace(v_min, v_max, steps=n_atoms))
        self.n = env.action_space.n
        dims = env.observation_space.shape
        self.network = nn.Sequential(
            nn.Conv2d(dims[0], 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, self.n * n_atoms),
        )

    def get_action(self, x, action=None):
        logits = self.network(x / 255.0)
        # probability mass function for each action
        pmfs = torch.softmax(logits.view(len(x), self.n, self.n_atoms), dim=2)
        q_values = (pmfs * self.atoms).sum(2)
        if action is None:
            action = torch.argmax(q_values, 1)
        return action, pmfs[torch.arange(len(x)), action]

    def get_action_and_value(self, x):
        logits = self.network(x / 255.0)
        # probability mass function for each action
        pmfs = torch.softmax(logits.view(len(x), self.n, self.n_atoms), dim=2)
        q_values = (pmfs * self.atoms).sum(2)
        action = torch.argmax(q_values, 1)
        return action, q_values[:, action], None, None
