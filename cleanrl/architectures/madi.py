import torch.nn as nn

from .common import NormalizeImg


class MaskerNet(nn.Module):
    def __init__(self, obs_shape, num_layers, num_filters):
        super().__init__()
        assert len(obs_shape) == 3
        self.img_size = obs_shape[-1]

        self.layers = nn.Sequential(NormalizeImg())

        for i in range(num_layers):
            if i == 0:
                in_filters = 1
            else:
                in_filters = num_filters
                self.layers.append(nn.ReLU())

            if i == num_layers - 1:
                out_filters = 1
            else:
                out_filters = num_filters

            self.layers.append(nn.Conv2d(in_filters, out_filters, 3, stride=1, padding=1, padding_mode='zeros'))

        self.layers.append(nn.Sigmoid())

        self.apply(weight_init)

    def forward(self, x):
        return x * self.layers(x.view(-1, 1, self.img_size, self.img_size)).view(x.shape)

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers"""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)