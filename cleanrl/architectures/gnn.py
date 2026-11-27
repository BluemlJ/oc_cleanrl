import torch
from torch import nn

from cleanrl.ppo_atari_oc_transformer import PPOAgent as TransformerPPOAgent

class GNNStack(nn.Module):
    def __init__(self, blocks, emb_dim, device):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.final_norm = nn.LayerNorm(emb_dim, device=device)

    def forward(self, x, src_key_padding_mask=None):
        for block in self.blocks:
            x = block(x, src_key_padding_mask=src_key_padding_mask)
        x = self.final_norm(x)
        if src_key_padding_mask is not None:
            x = x.masked_fill(src_key_padding_mask.unsqueeze(-1), 0.0)
        return x


def make_mlp(in_dim, out_dim, num_layers, device):
    assert num_layers >= 1
    layers = []
    for i in range(num_layers):
        layers.append(nn.Linear(in_dim if i == 0 else out_dim, out_dim, device=device))
        if i < num_layers - 1:           # no activation after the last layer
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class GNNLayer(nn.Module):
    def __init__(self, device, args):
        super().__init__()
        d = args.emb_dim
        # first edge layer split into receiver / sender halves (same function as Linear(2d, d))
        self.edge_in_i = nn.Linear(d, d, device=device)
        self.edge_in_j = nn.Linear(d, d, bias=False, device=device)
        rest = []
        for _ in range(args.num_edge_layers - 1):
            rest += [nn.ReLU(), nn.Linear(d, d, device=device)]
        self.edge_rest = nn.Sequential(*rest)   # empty Sequential = identity
        self.node_mlp = make_mlp(2 * d, d, args.num_node_layers, device)
        if args.gnn_pooling_type not in ('max', 'mean', 'sum', 'first'):
            raise NotImplementedError(args.gnn_pooling_type)
        self.pooling = args.gnn_pooling_type

    def _pool(self, msg, valid):
        # msg: [B, N, N, D], valid: broadcastable to [B, N, N, 1], True = use this sender
        if self.pooling == 'first':
            N = msg.shape[1]
            i = torch.arange(N, device=msg.device)
            return msg[:, i, (i == 0).long()]       # first sender other than i itself
        fill = float('-inf') if self.pooling == 'max' else 0.0
        msg = msg.masked_fill(~valid, fill)
        if self.pooling == 'max':
            return torch.nan_to_num(msg.max(dim=2).values, neginf=0.0)
        if self.pooling == 'sum':
            return msg.sum(dim=2)
        return msg.sum(dim=2) / valid.sum(dim=2).clamp(min=1)

    def forward(self, x, src_key_padding_mask=None):
        # x: [B, N, D]; src_key_padding_mask: [B, N], True = padding
        B, N, D = x.shape
        h = self.edge_in_i(x).unsqueeze(2) + self.edge_in_j(x).unsqueeze(1)  # [B, N, N, D]
        msg = self.edge_rest(h)                                               # [B, N, N, D]

        # valid = ~torch.eye(N, dtype=torch.bool, device=x.device)             # no self-messages
        valid = torch.ones(N, N, dtype=torch.bool, device=x.device)          # use self-messages
        if src_key_padding_mask is not None:
            valid = valid & ~src_key_padding_mask.unsqueeze(1)                # no padded senders
        valid = valid.unsqueeze(-1)                                           # [B or 1, N, N, 1]

        agg = self._pool(msg, valid)
        out = self.node_mlp(torch.cat([x, agg], dim=-1))
        if src_key_padding_mask is not None:
            out = out.masked_fill(src_key_padding_mask.unsqueeze(-1), 0.0)
        return out


class Residual(nn.Module):
    def __init__(self, layer, emb_dim, device):
        super().__init__()
        self.layer = layer
        self.norm = nn.LayerNorm(emb_dim, device=device)

    def forward(self, x, src_key_padding_mask=None):
        # post-norm (like nn.TransformerEncoderLayer's default): with pre-norm, the un-normalized
        # encoder output (~100x larger than a block's update) dominated the residual stream
        out = self.norm(x + self.layer(x, src_key_padding_mask=src_key_padding_mask))
        if src_key_padding_mask is not None:
            out = out.masked_fill(src_key_padding_mask.unsqueeze(-1), 0.0)
        return out


class GNNPPOAgent(TransformerPPOAgent):
    def __init__(self, envs, emb_dim, num_heads, num_blocks, masking, num_object_types, device, args):
        super().__init__(envs, emb_dim, num_heads, num_blocks, masking, num_object_types, device, args)

        self.transformer = GNNStack([Residual(GNNLayer(device, args), emb_dim, device) for _ in range(num_blocks)], emb_dim, device)
