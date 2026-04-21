import torch
from torch import nn

from .common import NormalizeImg
from .rainbow import NoisyLinear, F


class NoisyDuelingDistributionalNetwork(nn.Module):
    def __init__(self, env, n_atoms, v_min, v_max, n_actions):
        super().__init__()
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (n_atoms - 1)
        self.n_actions = n_actions
        self.register_buffer("support", torch.linspace(v_min, v_max, n_atoms))

        self.network = nn.Sequential(
            nn.Conv2d(env.observation_space.shape[0], 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        conv_output_size = 3136

        self.value_head = nn.Sequential(NoisyLinear(
            conv_output_size, 512), nn.ReLU(), NoisyLinear(512, n_atoms))

        self.advantage_head = nn.Sequential(
            NoisyLinear(conv_output_size, 512), nn.ReLU(
            ), NoisyLinear(512, n_atoms * self.n_actions)
        )

    def forward(self, x):
        h = self.network(x / 255.0)
        value = self.value_head(h).view(-1, 1, self.n_atoms)
        advantage = self.advantage_head(
            h).view(-1, self.n_actions, self.n_atoms)
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        q_dist = F.softmax(q_atoms, dim=2)
        return q_dist


    def draw_action(self, obs):
        q_dist = self.forward(obs)
        q_values = torch.sum(q_dist * self.support, dim=2)
        actions = torch.argmax(q_values, dim=1)
        return actions


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


class PPODefault(nn.Module):
    def __init__(self, envs, device, normalize=True):
        super().__init__()
        self.device = device

        dims = envs.observation_space.shape

        self.network = nn.Sequential(
            nn.Conv2d(dims[0], 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU()
        )

        if normalize:  # x / 255
            self.network.insert(0, NormalizeImg())

        self.network.append(nn.Flatten())

        # compute flatten size with a dummy forward
        # makes the agent applicable for any input image size
        with torch.no_grad():
            f = self.network(torch.zeros((1,) + dims))
            feat_dim = f.flatten().shape[0]

        self.network.append(nn.Linear(feat_dim, 512))
        self.network.append(nn.ReLU())

        self.actor = nn.Linear(512, envs.action_space.n)
        self.critic = nn.Linear(512, 1)

    def draw_action(self, state):
        hidden = self.network(state)
        logits = self.actor(hidden)
        return torch.argmax(logits)

    def features(self, x):
        return self.network(x)


def init_agent(env, ckpt, device):
    if "advantage_head.2.weight_mu" in ckpt["model_weights"]:
        args = ckpt["args"]
        agent = NoisyDuelingDistributionalNetwork(
            env,
            args["n_atoms"],
            args["v_min"],
            args["v_max"],
            env.action_space.n
        )
        agent.load_state_dict(ckpt["model_weights"])
        agent.eval()
        return agent

    elif "network.0.weight" in ckpt["model_weights"]:
        agent_class = PPOAgentNetwork
    elif "network.1.weight" in ckpt["model_weights"]:
        agent_class = PPODefault
    elif "features.0.weight" in ckpt["model_weights"]:
        agent_class = PPOAgentFeatures
    else:
        raise ValueError()

    agent = agent_class(env, device)
    agent.load_state_dict(ckpt["model_weights"])
    agent.eval()
    return agent