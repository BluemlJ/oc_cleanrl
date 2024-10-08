import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions.categorical import Categorical

from ocatari.core import OCAtari
from ocrltransformer.wrappers import OCWrapper
from torch.nn import TransformerEncoderLayer, TransformerEncoder

from vit_pytorch import SimpleViT
from vit_pytorch.mobile_vit import MobileViT


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


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
        self.device = device

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
    
    def predict(self, x, states=None, **_):
        with torch.no_grad():
            return np.argmax(self.actor(self.network(torch.Tensor(x).to(self.device))).cpu().numpy(), axis=1), states



class MobileVIT(nn.Module):
    def __init__(self, envs, emb_dim, num_heads, num_blocks, patch_size,
                 buffer_window_size, device):
        super().__init__()
        self.device = device
    
        self.network = nn.Sequential(
                MobileViT(
                    image_size=(84,84),
                    num_classes=emb_dim,
                    dims = [96, 120, 144],
                    channels = [4,4]
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
    
    def predict(self, x, states=None, **_):
        with torch.no_grad():
            return np.argmax(self.actor(self.network(torch.Tensor(x).to(self.device))).cpu().numpy(), axis=1), states



class MobileViT2(nn.Module):
    def __init__(self, envs, emb_dim, num_heads, num_blocks, patch_size,
                 buffer_window_size, device):
        super().__init__()
        self.device = device
    
        self.network = MobileViT(
        image_size = (84, 84),
        dims = [96, 120, 144],
        channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384],
        num_classes = envs.action_space.n
        )

    def get_value(self, x):
        return (self.network(x))

    def get_action_and_value(self, x, action=None):
        logits = self.network(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

    def predict(self, x, states=None, **_):
        with torch.no_grad():
            return np.argmax(self.actor(self.network(torch.Tensor(x).to(self.device))).cpu().numpy(), axis=1), states



class SimpleViT2(nn.Module):
    def __init__(self, envs, emb_dim, num_heads, num_blocks, patch_size,
                 buffer_window_size, device):
        super().__init__()
        self.device = device

        self.network = SimpleViT(
        image_size = 84,
        patch_size=patch_size,
        channels=buffer_window_size,
        num_classes=envs.action_space.n,
        dim=emb_dim,
        depth=num_blocks,
        heads=num_heads,
        mlp_dim=emb_dim,
        )

    def get_value(self, x):
        return (self.network(x))

    def get_action_and_value(self, x, action=None):
        return self.get_value(x),0,0,0

    def predict(self, x, states=None, **_):
        with torch.no_grad():
            return np.argmax(self.actor(self.network(torch.Tensor(x).to(self.device))).cpu().numpy(), axis=1), states

