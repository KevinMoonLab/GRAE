"""Torch modules."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearBlock(nn.Module):
    """FC layer with Relu activation."""

    def __init__(self, in_dim, out_dim):
        """Init.

        Args:
            in_dim(int): Input dimension.
            out_dim(int): Output dimension
        """
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        """Forward pass.

        Args:
            x(torch.Tensor): Input data.

        Returns:
            torch.Tensor: Activations.

        """
        return F.relu(self.linear(x))


class MLP(nn.Sequential):
    """Sequence of FC layers with Relu activations.

    No activation on last layer, unless sigmoid is requested."""

    def __init__(self, dim_list, sigmoid=False):
        """Init.

        Args:
            dim_list(List[int]): List of dimensions. Ex: [200, 100, 50] will create two layers (200x100 followed by
            100x50).
        """
        # Activations on all layers except last one
        modules = [LinearBlock(dim_list[i - 1], dim_list[i]) for i in range(1, len(dim_list) - 1)]
        modules.append(nn.Linear(dim_list[-2], dim_list[-1]))

        if sigmoid:
            modules.append(nn.Sigmoid())

        super().__init__(*modules)



class AutoencoderModule(nn.Module):
    """Vanilla Autoencoder torch module"""

    def __init__(self, input_dim, hidden_dims, z_dim, noise=0, vae=False, sigmoid=False):
        """Init.

        Args:
            input_dim(int): Dimension of the input data.
            hidden_dims(List[int]): List of hidden dimensions. Do not include dimensions of the input layer and the
            bottleneck. See MLP for example.
            z_dim(int): Bottleneck dimension.
            noise(float): Variance of the gaussian noise applied to the latent space before reconstruction.
            vae(bool): Make this architecture a VAE. Uses an isotropic Gaussian with identity covariance matrix as the
            prior.
            sigmoid(bool): Apply sigmoid to the output.
        """
        super().__init__()
        self.vae = vae

        # Double the size of the latent space if vae to model both mu and logvar
        full_list = [input_dim] + list(hidden_dims) + [z_dim * 2 if vae else z_dim]

        self.encoder = MLP(dim_list=full_list)

        full_list.reverse()  # Use reversed architecture for decoder
        full_list[0] = z_dim

        self.decoder = MLP(dim_list=full_list, sigmoid=sigmoid)
        self.noise = noise

    def forward(self, x):
        """Forward pass.

        Args:
            x(torch.Tensor): Input data.

        Returns:
            tuple:
                torch.Tensor: Reconstructions
                torch.Tensor: Embedding (latent space coordinates)

        """
        z = self.encoder(x)

        # Old idea to inject noise in latent space. Currently not used.
        if self.noise > 0:
            z_decoder = z + self.noise * torch.randn_like(z)
        else:
            z_decoder = z

        if self.vae:
            mu, logvar = z.chunk(2, dim=-1)

            # Reparametrization trick
            if self.training:
                z_decoder = mu + torch.exp(logvar / 2.) * torch.randn_like(logvar)
            else:
                z_decoder = mu

        output = self.decoder(z_decoder)

        # Standard Autoencoder forward pass
        # Note : will still return mu and logvar as a single tensor for compatibility with other classes
        return output, z


# Convolution architecture
class DownConvBlock(nn.Module):
    """Convolutional block.

    3x3 kernel with 1 padding, Max pooling and Relu activations. Channels must be specified by user.

    """
    def __init__(self, in_channels, out_channels):
        """Init.

        Args:
            in_channels(int): Input channels.
            out_channels(int): Output channels.
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.max = nn.MaxPool2d(2)

    def forward(self, x):
        """Forward pass.

        Args:
            x(torch.Tensor): Input data.

        Returns:
            torch.Tensor: Activations.

        """
        x = self.conv(x)
        x = F.relu(x)
        x = self.max(x)

        return x


class UpConvBlock(nn.Module):
    """Transpose convolutional block to upscale input.

    2x2 Transpoe convolution followed by a convolutional layer with
    3x3 kernel with 1 padding, Max pooling and Relu activations. Channels must be specified by user.
    """

    def __init__(self, in_channels, out_channels):
        """Init.

        Args:
            in_channels(int): Input channels.
            out_channels(int): Output channels.
        """
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        """Forward pass.

        Args:
            x(torch.Tensor): Input data.

        Returns:
            torch.Tensor: Activations.

        """
        x = self.up(x)
        x = self.conv(x)
        x = F.relu(x)

        return x


class LastConv(UpConvBlock):
    """Add one convolution to UpConvBlock with no activation and kernel_size = 1.

    Used as the output layer in the convolutional AE architecture."""

    def __init__(self, in_channels, out_channels):
        """Init.

        Args:
            in_channels(int): Input channels.
            out_channels(int): Output channels.
        """
        super().__init__(in_channels, in_channels)
        self.conv_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """Forward pass.

        Args:
            x(torch.Tensor): Input data.

        Returns:
            torch.Tensor: Activations.

        """
        x = super().forward(x)
        x = self.conv_2(x)

        return x


class ConvEncoder(nn.Module):
    """Convolutional encoder for images datasets.

    Convolutions (with 3x3 kernel size, see DownConvBlock for details) followed by a FC network."""
    def __init__(self, H, W, input_channel, channel_list, hidden_dims, z_dim):
        """Init.

        Args:
            H(int): Height of the input data.
            W(int): Width of the input data
            input_channel(int): Number of channels in the input data. Typically 1 for grayscale and 3 for RGB.
            channel_list(List[int]): List of channels. Determines the number of convolutional layers and associated
            channels.  ex: [128, 64, 32] defines one layer with in_channels=128 and out_channels=64 followed by a
            second one with in_channels=64 and out_channels=32.
            hidden_dims(List[int]): List of hidden dimensions for the FC network after the convolutions.
            Do not include dimensions of the input layer and the bottleneck. See MLP for example.
            z_dim(int): Dimension of the bottleneck.
        """
        super().__init__()

        channels = [input_channel] + channel_list
        modules = list()

        for i in range(1, len(channels)):
            modules.append(DownConvBlock(in_channels=channels[i - 1], out_channels=channels[i]))

        self.conv = nn.Sequential(*modules)

        factor = 2 ** len(channel_list)

        self.fc_size = int(channel_list[-1] * H / factor * W / factor)  # Compute size of FC input

        mlp_dim = [self.fc_size] + hidden_dims + [z_dim]

        self.linear = MLP(mlp_dim)

    def forward(self, x):
        """Forward pass.

        Args:
            x(torch.Tensor): Input data.

        Returns:
            torch.Tensor: Activations.

        """
        x = self.conv(x)
        x = x.view(-1, self.fc_size)
        x = self.linear(x)

        return x


class ConvDecoder(nn.Module):
    """Convolutional decoder for images datasets.

    FC architecture followed by upscaling convolutions.
    Note that last layer uses a 1x1 kernel with no activations. See UpConvBlock and LastConv for details."""
    def __init__(self, H, W, input_channel, channel_list, hidden_dims, z_dim, sigmoid):
        """Init.

        Args:
            H(int): Height of the reconstructed data.
            W(int): Width of the reconstructed data
            input_channel(int): Number of channels in the reconstructed data. Typically 1 for grayscale and 3 for RGB.
            channel_list(List[int]): List of channels. Determines the number of UpConvBlock and associated
            channels.  ex: [32, 64, 128] defines one layer with in_channels=32 and out_channels=64 followed by a
            second one with in_channels=64 and out_channels=128.
            hidden_dims(List[int]): List of hidden dimensions for the FC network before the convolutions.
            Do not include dimensions of the input layer and the bottleneck. See MLP for example.
            z_dim(int): Dimension of the bottleneck.
            sigmoid(bool) : Apply sigmoid to output.
        """
        super().__init__()
        self.H = H
        self.W = W

        self.factor = 2 ** len(channel_list)

        fc_size = int(channel_list[0] * H / self.factor * W / self.factor)

        mlp_dim = [z_dim] + hidden_dims + [fc_size]

        self.linear = MLP(mlp_dim)

        channels = channel_list
        modules = list()

        for i in range(1, len(channels)):
            modules.append(UpConvBlock(in_channels=channels[i - 1], out_channels=channels[i]))

        modules.append(LastConv(in_channels=channels[-1], out_channels=input_channel))

        if sigmoid:
            modules.append(nn.Sigmoid())

        self.conv = nn.Sequential(*modules)
        self.first_channel = channel_list[0]

    def forward(self, x):
        """Forward pass.

        Args:
            x(torch.Tensor): Input data.

        Returns:
            torch.Tensor: Activations.

        """
        x = self.linear(x)
        x = x.view(-1, self.first_channel, self.H // self.factor, self.W // self.factor)
        x = self.conv(x)

        return x


class ConvAutoencoderModule(nn.Module):
    """Autoencoder with convolutions for image datasets."""
    def __init__(self, H, W, input_channel, channel_list, hidden_dims, z_dim, noise, vae=False, sigmoid=False):
        """Init. Arguments specify the architecture of the encoder. Decoder will use the reverse architecture.

        Args:
            H(int): Height of the input data.
            W(int): Width of the input data
            input_channel(int): Number of channels in the input data. Typically 1 for grayscale and 3 for RGB.
            channel_list(List[int]): List of channels. Determines the number of convolutional layers and associated
            channels.  ex: [128, 64, 32] defines one layer with in_channels=128 and out_channels=64 followed by a
            second one with in_channels=64 and out_channels=32.
            hidden_dims(List[int]): List of hidden dimensions for the FC network after the convolutions.
            Do not include dimensions of the input layer and the bottleneck. See MLP for example.
            z_dim(int): Dimension of the bottleneck.
            noise(float): Variance of the gaussian noise applied to the latent space before reconstruction.
            vae(bool): Make this architecture a VAE. Uses an isotropic Gaussian with identity covariance matrix as the
            prior.
            sigmoid(bool): Apply sigmoid to the output.
        """
        super().__init__()
        self.vae = vae

        # Double size of encoder output if using a VAE to model both mu and logvar
        self.encoder = ConvEncoder(H, W, input_channel, channel_list, hidden_dims, z_dim * 2 if self.vae else z_dim)
        channel_list.reverse()
        hidden_dims.reverse()
        self.decoder = ConvDecoder(H, W, input_channel, channel_list, hidden_dims, z_dim, sigmoid)
        self.noise = noise

    def forward(self, x):
        """Forward pass.

        Args:
            x(torch.Tensor): Input data.

        Returns:
            tuple:
                torch.Tensor: Reconstructions
                torch.Tensor: Embedding (latent space coordinates)
        """
        # Same forward pass as standard autoencoder
        return AutoencoderModule.forward(self, x)
