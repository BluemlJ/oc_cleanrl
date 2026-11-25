import torch
from torch import nn

from ppo_atari_oc_transformer import PPOAgent as TransformerPPOAgent

class SeqLayer(nn.Sequential):
    def forward(self, x, src_key_padding_mask=None):
        for module in self:
            x = module(x, src_key_padding_mask)
        return x


class DeepSetsLayer(nn.Module):
    def __init__(self, device, args):
        super(DeepSetsLayer, self).__init__()

        self.lamb = nn.Parameter(torch.normal(torch.zeros(1), torch.ones(1) * 0.2)).to(device)
        self.gamma = nn.Parameter(torch.normal(torch.zeros(1), torch.ones(1) * 0.2).to(device))
        self.activation = nn.ReLU()
        self.device = device

        if args.ds_pooling_type == 'max':
            self.pool = lambda x: torch.max(x, dim=1, keepdim=True)[0]
        elif args.ds_pooling_type == 'mean':
            self.pool = lambda x: torch.mean(x, dim=1, keepdim=True)
        elif args.ds_pooling_type == 'first':
            self.pool = lambda x: x[:, 0, :]
        else:
            raise NotImplementedError


    def forward(self, x, src_key_padding_mask=None):
        if src_key_padding_mask is not None:
            x0 = x * src_key_padding_mask.unsqueeze(-1)
            _max = self.pool(x0)
            return self.activation((self.lamb * x0 + self.gamma * _max) * src_key_padding_mask.unsqueeze(-1))
        return self.activation((self.lamb * x + self.gamma * self.pool(x)))


class DeepSetsPPOAgent(TransformerPPOAgent):
    def __init__(self, envs, emb_dim, num_heads, num_blocks, masking, num_object_types, device, args):
        super().__init__(envs, emb_dim, num_heads, num_blocks, masking, num_object_types, device, args)

        self.transformer = SeqLayer(*[DeepSetsLayer(device, args) for _ in range(num_blocks)])