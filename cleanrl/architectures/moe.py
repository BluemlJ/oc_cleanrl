import torch
import torch.nn as nn
import torch.functional as F
from torch.distributions.categorical import Categorical

from typing import Optional

from .common import Predictor, layer_init


class MoEAgent(Predictor):
    """
    Policy/value network that gates over multiple expert policies (optionally
    augmented with downsampled pixels).
    """

    def __init__(
        self,
        envs,
        device,
        layer_dims=(128, 64),
        weighted_sum: bool = True,
        include_policies: bool = True,
        top_k: Optional[int] = None,
        include_value_in_policy_input: bool = False,
    ):
        super().__init__()
        base_env = envs
        while hasattr(base_env, "venv"):
            base_env = base_env.venv
        # metadata provided by MoEWrapper
        self.action_dim = base_env.get_attr("action_dim")[0]
        self.num_experts = base_env.get_attr("num_experts")[0]
        self.expert_feature_dim = base_env.get_attr("expert_feature_dim")[0]
        self.raw_obs_dim = base_env.get_attr("raw_obs_dim")[0]
        self.encoded_raw_hw = base_env.get_attr("encoded_raw_hw")[0]

        # option to drop expert value from the policy branch input
        if include_value_in_policy_input:
            self.policy_input_dim = self.expert_feature_dim
        else:
            self.policy_input_dim = self.expert_feature_dim - self.num_experts
        self.per_expert_dim = self.policy_input_dim // self.num_experts

        self.include_policies = include_policies
        self.include_pixels = self.raw_obs_dim > 0 and self.encoded_raw_hw is not None
        self.top_k = top_k
        assert self.include_policies or self.include_pixels

        # policy branch
        if include_policies:
            policy_hidden_dim = layer_dims[0] if layer_dims else 128
            self.policy_branch = nn.Sequential(
                layer_init(nn.Linear(self.policy_input_dim, policy_hidden_dim)),
                nn.ReLU(),
            )
        else:
            policy_hidden_dim = 0

        # raw pixel branch (optional)
        if self.include_pixels:
            raw_h, raw_w = self.encoded_raw_hw
            self.raw_branch = nn.Sequential(
                layer_init(nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1)),
                nn.ReLU(),
                layer_init(nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)),
                nn.ReLU(),
                nn.Flatten(),
            )
            conv_out_h = raw_h // 4
            conv_out_w = raw_w // 4
            conv_out_dim = 16 * conv_out_h * conv_out_w
            raw_hidden_dim = max(policy_hidden_dim, 16)
            self.raw_project = nn.Sequential(
                layer_init(nn.Linear(conv_out_dim, raw_hidden_dim)),
                nn.ReLU(),
            )
        else:
            self.raw_branch = None
            self.raw_project = None
            raw_hidden_dim = 0

        # fusion MLP
        fusion_input_dim = policy_hidden_dim + raw_hidden_dim
        fusion_layers: list[nn.Module] = []
        current_dim = fusion_input_dim
        for hidden_dim in layer_dims[1:]:
            fusion_layers.append(layer_init(nn.Linear(current_dim, hidden_dim)))
            fusion_layers.append(nn.ReLU())
            current_dim = hidden_dim
        self.fusion_network = nn.Sequential(*fusion_layers) if fusion_layers else nn.Identity()
        feature_dim = current_dim if fusion_layers else fusion_input_dim

        if weighted_sum:
            output_dim = self.num_experts  # gating distribution
            self.weighted_sum = True
        else:
            output_dim = self.action_dim  # direct policy logits
            self.weighted_sum = False

        self.actor = layer_init(nn.Linear(feature_dim, output_dim), std=0.01)
        self.critic = layer_init(nn.Linear(feature_dim, 1), std=1)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.include_policies:
            policy_value_part = x[:, : self.expert_feature_dim]
            experts_and_values = policy_value_part.reshape(
                x.size(0), self.num_experts, self.action_dim + 1
            )
            policy_only = torch.clamp(experts_and_values[..., : self.per_expert_dim], min=1e-8)
            features = self.policy_branch(policy_only.reshape(x.size(0), -1))

        if self.include_pixels:
            raw_part = x[:, self.expert_feature_dim :]
            raw_h, raw_w = self.encoded_raw_hw
            raw_img = raw_part.view(x.size(0), 1, raw_h, raw_w)
            raw_features = self.raw_branch(raw_img)
            raw_features = self.raw_project(raw_features)
            if self.include_policies:
                features = torch.cat((features, raw_features), dim=1)
            else:
                features = raw_features

        return self.fusion_network(features)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self._forward_features(x)
        return self.critic(hidden).squeeze(-1)

    def _direct_distribution(self, hidden: torch.Tensor) -> Categorical:
        logits = self.actor(hidden)
        return Categorical(logits=logits)

    def _weighted_distribution(self, hidden: torch.Tensor, x: torch.Tensor, tau: float = 0.0, temperature: float = 0.0, top_k: bool = False):
        # gating
        weights = torch.softmax(self.actor(hidden), dim=1)

        # save predicted weights for logging and entropy loss
        predicted_weights_categorical = Categorical(probs=weights)

        if tau > 0.0:
            weights = weights + tau * torch.rand_like(weights)
            weights = weights.clamp_min(1e-8)
            weights = weights / weights.sum(dim=1, keepdim=True)

        policy_value_part = x[:, : self.expert_feature_dim]
        experts_and_values = policy_value_part.reshape(x.size(0), self.num_experts, self.action_dim + 1)
        values = experts_and_values[..., -1]
        experts = torch.clamp(experts_and_values[..., :-1], min=1e-8)
        experts = experts / experts.sum(dim=2, keepdim=True)

        if top_k and self.top_k is not None:
            weights, indices = torch.topk(weights, self.top_k, dim=1)
            experts = torch.gather(experts, 1, indices.unsqueeze(-1).expand(-1, -1, experts.size(-1)))
            values = torch.gather(values, 1, indices)
            weights = weights / weights.sum(dim=1, keepdim=True)

        mixed_probs = torch.sum(weights.unsqueeze(-1) * experts, dim=1)
        mixed_values = values.mean(dim=1)

        # Add temperature
        if temperature > 0:
            mixed_probs = mixed_probs ** temperature
            mixed_probs /= mixed_probs.sum(dim=1, keepdim=True)

        return Categorical(probs=mixed_probs.to(hidden.device)), mixed_values.detach(), predicted_weights_categorical

    @staticmethod
    def current_value(global_step: int, start_value: float, end_value: float, decay_steps: int) -> float:
        if decay_steps <= 0:
            return end_value
        progress = min(1.0, float(global_step) / float(decay_steps))
        return start_value + progress * (end_value - start_value)

    def get_mixture_weights(self, x: torch.Tensor) -> torch.Tensor:
        assert self.weighted_sum, "Mixture weights only exist when weighted_sum=True"
        with torch.no_grad():
            hidden = self._forward_features(x)
            weights = torch.softmax(self.actor(hidden), dim=1)
        return weights

    def get_action_and_value(self, x: torch.Tensor, action=None, tau: float = 0.0, temperature: float = 0.0, top_k: bool = False):
        hidden = self._forward_features(x)
        if self.weighted_sum:
            categorical, _, predicted_weights_categorical = self._weighted_distribution(hidden, x, tau, temperature, top_k=top_k)
        else:
            categorical = self._direct_distribution(hidden)
            predicted_weights_categorical = None
        if action is None:
            action = categorical.sample()
        value = self.critic(hidden).squeeze(-1)
        return action, categorical.log_prob(action), predicted_weights_categorical.entropy(), value, predicted_weights_categorical.probs
