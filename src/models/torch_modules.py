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


# Convolution architecture
class DownConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.max = nn.MaxPool2d(2)

    def forward(self, x):
        return self.max(self.conv(x))


class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))

class ConvEncoder(nn.Module):
    def __init__(self, H, W, input_channel, channel_list, hidden_dims, z_dim):
        super().__init__()

        channels = [input_channel] + channel_list
        modules = list()

        for i in range(1, len(channels)):
            modules.append(DownConvBlock(in_channels=channels[i - 1], out_channels=channels[i]))

        self.conv = nn.Sequential(*modules)

        factor = 2 ** len(channel_list)

        self.fc_size = int(channel_list[-1] * H/factor * W/factor)

        mlp_dim = [self.fc_size] + hidden_dims + [z_dim]

        self.linear = MLP(mlp_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, self.fc_size)
        x = self.linear(x)

        return x


class ConvDecoder(nn.Module):
    def __init__(self, H, W, input_channel, channel_list, hidden_dims, z_dim):
        super().__init__()
        self.H = H
        self.W = W

        self.factor = 2 ** len(channel_list)

        fc_size = int(channel_list[0] * H / self.factor * W / self.factor)

        mlp_dim = [z_dim] + hidden_dims + [fc_size]

        self.linear = MLP(mlp_dim)

        channels = channel_list + [input_channel]
        modules = list()

        for i in range(1, len(channels)):
            modules.append(UpConvBlock(in_channels=channels[i - 1], out_channels=channels[i]))

        self.conv = nn.Sequential(*modules)
        self.first_channel = channel_list[0]

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, self.first_channel, self.H//self.factor, self.W//self.factor)
        x = self.conv(x)

        return x

class ConvAutoencoderModule(nn.Module):
    def __init__(self, H, W, input_channel, channel_list, hidden_dims, z_dim):
        super().__init__()

        self.encoder = ConvEncoder(H, W, input_channel, channel_list, hidden_dims, z_dim)
        channel_list.reverse()
        hidden_dims.reverse()
        self.decoder = ConvDecoder(H, W, input_channel, channel_list, hidden_dims, z_dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z
