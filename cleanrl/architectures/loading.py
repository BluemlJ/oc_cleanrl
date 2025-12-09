import torch
from torch import nn

from .ppo import PPODefault

class PPOAgentFeatures(nn.Module):
    def __init__(self, env, device, normalize=True):
        super().__init__()
        self.device = device
        self.normalize = normalize

        dims = env.observation_space.shape
        self.features = nn.Sequential(
            nn.Identity(),
            nn.Conv2d(dims[0], 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
        )

        self.actor = nn.Linear(512, env.action_space.n)
        self.critic = nn.Linear(512, 1)

    def draw_action(self, state):
        if self.normalize:
            state = state / 255
        hidden = self.features(state)
        logits = self.actor(hidden)
        return torch.argmax(logits)

class PPOAgentNetwork(nn.Module):
    def __init__(self, env, device, normalize=True):
        super().__init__()
        self.device = device
        self.normalize = normalize

        dims = env.observation_space.shape
        self.network = nn.Sequential(
            nn.Conv2d(dims[0], 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
        )

        self.actor = nn.Linear(512, env.action_space.n)
        self.critic = nn.Linear(512, 1)

    def draw_action(self, state):
        if self.normalize:
            state = state / 255
        hidden = self.network(state)
        logits = self.actor(hidden)
        return torch.argmax(logits)

    def features(self, x):
        return self.network(x)


def init_agent(env, ckpt, device):
    if "network.0.weight" in ckpt["model_weights"]:
        agent_class = PPOAgentNetwork
    elif "features.1.weight" in ckpt["model_weights"]:
        agent_class = PPODefault
    elif "features.0.weight" in ckpt["model_weights"]:
        agent_class = PPOAgentFeatures
    else:
        raise ValueError()

    agent = agent_class(env, device)
    agent.load_state_dict(ckpt["model_weights"])
    return agent