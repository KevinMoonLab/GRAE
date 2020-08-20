"""Torch modules."""
import torch.nn as nn
import torch.nn.functional as F


class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return F.relu(self.linear(x))


class MLP(nn.Sequential):
    def __init__(self, dim_list):
        # Activations on all layers except last one
        modules = [LinearBlock(dim_list[i - 1], dim_list[i]) for i in range(1, len(dim_list) - 1)]
        modules.append(nn.Linear(dim_list[-2], dim_list[-1]))

        super().__init__(*modules)


class AutoencoderModule(nn.Module):
    def __init__(self, input_dim, hidden_dims, z_dim):
        super().__init__()

        full_list = [input_dim] + list(hidden_dims) + [z_dim]

        self.encoder = MLP(dim_list=full_list)
        full_list.reverse()
        self.decoder = MLP(dim_list=full_list)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

