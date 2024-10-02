import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions.categorical import Categorical

from ocatari.core import OCAtari
from ocrltransformer.wrappers import OCWrapper
from torch.nn import TransformerEncoderLayer, TransformerEncoder

from vit_pytorch import SimpleViT


class OCTransformer(nn.Module):
    def __init__(self, envs, emb_dim, num_heads, num_blocks, device):
        super().__init__()

        self.device = device
        dims = envs.observation_space.feature_space.shape

        encoder_layer = TransformerEncoderLayer(emb_dim, num_heads,
                                                emb_dim, device=device,
                                                dropout=0.1, batch_first=True)

        self.network = nn.Sequential(
            nn.Linear(dims[1], emb_dim, device=device),
            TransformerEncoder(encoder_layer, num_blocks),
            nn.Flatten(),
        )
        self.actor = layer_init(nn.Linear(dims[0] * emb_dim, envs.action_space.n, device=device), std=0.01)
        self.critic = layer_init(nn.Linear(dims[0] * emb_dim, 1, device=device), std=1)

    def get_value(self, x):
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

    def predict(self, x, states=None, **_):
        with torch.no_grad():
            return np.argmax(self.actor(self.network(torch.Tensor(x).to(self.device))).cpu().numpy(), axis=1), states


class VIT(nn.Module):
    def __init__(self, envs, emb_dim, num_heads, num_blocks, patch_size,
                 buffer_window_size, device):
        super().__init__()

        self.network = nn.Sequential(
            SimpleViT(
                image_size=84,
                patch_size=patch_size,
                channels=buffer_window_size,
                num_classes=emb_dim,
                dim=emb_dim,
                depth=num_blocks,
                heads=num_heads,
                mlp_dim=emb_dim,
            ).to(device),
            nn.Flatten(),
        )
        self.actor = layer_init(nn.Linear(emb_dim, envs.action_space.n, device=device), std=0.01)
        self.critic = layer_init(nn.Linear(emb_dim, 1, device=device), std=1)

    def get_value(self, x):
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)
