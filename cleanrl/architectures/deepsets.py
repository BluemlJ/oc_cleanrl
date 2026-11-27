import torch
from torch import nn

from cleanrl.ppo_atari_oc_transformer import PPOAgent as TransformerPPOAgent


def masked_pool(x, keep, kind, keepdim=False):
    # x: [B, N, D]; keep: [B, N, 1], True = real object
    if kind == 'max':
        out = x.masked_fill(~keep, float('-inf')).max(dim=1, keepdim=keepdim).values
        return torch.nan_to_num(out, neginf=0.0)
    if kind == 'sum':
        return (x * keep).sum(dim=1, keepdim=keepdim)
    if kind == 'mean':
        return (x * keep).sum(dim=1, keepdim=keepdim) / keep.sum(dim=1, keepdim=keepdim).clamp(min=1)
    if kind == 'first':
        return x[:, :1] if keepdim else x[:, 0]
    raise NotImplementedError(kind)


class DeepSetsLayer(nn.Module):
    """Permutation-equivariant layer (Zaheer et al., 2017): relu(x Λ + pool(x) Γ)."""

    def __init__(self, device, args):
        super().__init__()
        d = args.emb_dim
        self.lamb = nn.Linear(d, d, device=device)
        self.gamma = nn.Linear(d, d, bias=False, device=device)
        self.activation = nn.ReLU()
        self.pooling = args.ds_pooling_type
        masked_pool(torch.zeros(1, 1, 1), torch.ones(1, 1, 1, dtype=torch.bool), self.pooling)  # validate

    def forward(self, x, keep):
        return self.activation(self.lamb(x) + self.gamma(masked_pool(x, keep, self.pooling, keepdim=True)))


class Residual(nn.Module):
    # post-norm, like nn.TransformerEncoderLayer and the GNN blocks
    def __init__(self, layer, emb_dim, device):
        super().__init__()
        self.layer = layer
        self.norm = nn.LayerNorm(emb_dim, device=device)

    def forward(self, x, keep):
        return self.norm(x + self.layer(x, keep)) * keep


class DeepSetsStack(nn.Module):
    """num_blocks equivariant blocks followed by a LayerNorm. With num_blocks = 0 this is just a
    per-object LayerNorm, i.e. the plain DeepSets phi (encoder [+ network]) -> pool -> rho."""

    def __init__(self, blocks, emb_dim, device):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.final_norm = nn.LayerNorm(emb_dim, device=device)

    def forward(self, x, keep):
        for block in self.blocks:
            x = block(x, keep)
        return self.final_norm(x) * keep


class DeepSetsPPOAgent(TransformerPPOAgent):
    """
    phi:  encoder (args.encoder_dims) -> num_blocks equivariant layers -> LayerNorm
    pool: args.pooling_type over real objects only (padding is masked out)
    rho:  network (args.num_post_layers x [Linear, ReLU]) on the pooled vector, then actor / critic heads
    """

    def __init__(self, envs, emb_dim, num_heads, num_blocks, masking, num_object_types, device, args):
        super().__init__(envs, emb_dim, num_heads, num_blocks, masking, num_object_types, device, args)

        self.transformer = DeepSetsStack(
            [Residual(DeepSetsLayer(device, args), emb_dim, device) for _ in range(num_blocks)],
            emb_dim, device,
        )
        self.readout = args.pooling_type
        self.use_mask = masking
        self.forward = self.deepsets_forward

    def deepsets_forward(self, x):
        if self.use_mask:
            keep = x[..., :self.num_object_types].sum(dim=-1, dtype=bool)
            keep[(~keep).all(-1)] = True  # no objects: treat all slots as real, like the transformer
        else:
            keep = torch.ones(x.shape[:-1], dtype=torch.bool, device=x.device)
        keep = keep.unsqueeze(-1)

        h = self.transformer(self.encoder(x), keep)
        return self.network(masked_pool(h, keep, self.readout))
