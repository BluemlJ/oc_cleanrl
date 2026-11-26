import torch
from torch import nn

from cleanrl.ppo_atari_oc_transformer import PPOAgent as TransformerPPOAgent

class SeqLayer(nn.Sequential):
    def forward(self, x, src_key_padding_mask=None):
        for module in self:
            x = module(x, src_key_padding_mask)
        return x


def pairwise_concat(x):
    """
    x: [B, N, F]  (envs, objects, features)
    returns: [B, N, N-1, 2F]  where out[b, i, k] = cat(x[b, i], x[b, j_k]) for all j != i
    """
    B, N, F = x.shape
    x_i = x.unsqueeze(2).expand(B, N, N, F)  # receiver i, repeated along j
    x_j = x.unsqueeze(1).expand(B, N, N, F)  # sender j, repeated along i
    pairs = torch.cat([x_i, x_j], dim=-1)  # [B, N, N, 2F]

    mask = ~torch.eye(N, dtype=torch.bool,
                      device=x.device)  # remove i == j
    return pairs[:, mask].view(B, N, N - 1, 2 * F)


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
        self.edge_mlp = make_mlp(2 * d, d, args.num_edge_layers, device)
        self.node_mlp = make_mlp(2 * d, d, args.num_node_layers, device)
        if args.gnn_pooling_type not in ('max', 'mean', 'sum'):
            raise NotImplementedError(args.gnn_pooling_type)
        self.pooling = args.gnn_pooling_type

    def _pool(self, msg, valid=None):
        # msg: [B, N, N-1, D], valid: [B, N, N-1, 1] bool (True = real sender) or None
        if valid is not None:
            fill = float('-inf') if self.pooling == 'max' else 0.0
            msg = msg.masked_fill(~valid, fill)
        if self.pooling == 'max':
            # objects with no valid neighbors would get -inf -> set to 0
            return torch.nan_to_num(msg.max(dim=2).values, neginf=0.0)
        if self.pooling == 'sum':
            return msg.sum(dim=2)
        if valid is None:
            return msg.mean(dim=2)
        return msg.sum(dim=2) / valid.sum(dim=2).clamp(min=1)

    def forward(self, x, src_key_padding_mask=None):
        # x: [B, N, D]; src_key_padding_mask: [B, N], True = padding
        B, N, D = x.shape
        msg = self.edge_mlp(pairwise_concat(x))           # [B, N, N-1, D]

        valid = None
        if src_key_padding_mask is not None:
            keep = ~src_key_padding_mask                   # True = real object
            off_diag = ~torch.eye(N, dtype=torch.bool, device=x.device)
            # sender_ok[b, i, k] = keep[b, j_k], same ordering as pairwise_concat
            sender_ok = keep.unsqueeze(1).expand(B, N, N)[:, off_diag].view(B, N, N - 1)
            valid = sender_ok.unsqueeze(-1)

        agg = self._pool(msg, valid)                       # [B, N, D]
        out = self.node_mlp(torch.cat([x, agg], dim=-1))   # [B, N, D]

        if src_key_padding_mask is not None:
            out = out.masked_fill(src_key_padding_mask.unsqueeze(-1), 0.0)
        return out


class GNNPPOAgent(TransformerPPOAgent):
    def __init__(self, envs, emb_dim, num_heads, num_blocks, masking, num_object_types, device, args):
        super().__init__(envs, emb_dim, num_heads, num_blocks, masking, num_object_types, device, args)

        self.transformer = SeqLayer(*[GNNLayer(device, args) for _ in range(num_blocks)])
