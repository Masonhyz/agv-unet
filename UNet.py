import torch
import torch.nn as nn


def double_conv_block(in_channels, out_channels):
    """
    a double convolution block with inplace ReLU activation
    :param in_channels: int, usually multiples of 2
    :param out_channels: int, usually multiples of 2
    :return: a double convolution block
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )


def downsample_block(in_channels, out_channels, dropout):
    """
    a downsampling block in the contractive path of a UNet
    :param in_channels: int, usually half as much as out_channels
    :param out_channels: int, usually twice as much as in_channels
    :param dropout: dropout
    :return: a downsampling block
    """
    return nn.Sequential(
        nn.MaxPool2d(2),
        nn.Dropout(dropout),
        double_conv_block(in_channels, out_channels)
    )


def upsample_block(in_channels, out_channels, dropout):
    """
    an upsampling block in the expansive path of a UNet
    :param in_channels: int, usually multiple of two
    :param out_channels: int, usually half as in_channels
    :param dropout: dropout
    :return: the upsampling block
    """
    return nn.Sequential(
        nn.Dropout(dropout),
        double_conv_block(in_channels * 2, in_channels),
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2,
                           padding=1, output_padding=1))


class Upsample_block(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super(Upsample_block, self).__init__()

        self.drop = nn.Dropout(dropout)
        self.conv = double_conv_block(in_channels * 2, in_channels)
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3,
                                     stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = self.drop(x)
        x = self.conv(x)
        y = self.up(x)
        return x, y


class UNet(nn.Module):
    """
    Standard UNet class
    """

    def __init__(self, cfg):
        """
        initializer of UNet
        :param cfg: the model configurations
        """
        super(UNet, self).__init__()
        c = cfg["channel"]
        dropout_rate = cfg["dropout"]

        self.down0 = double_conv_block(3, c)
        self.down1 = downsample_block(c, c * 2, dropout_rate)
        self.down2 = downsample_block(c * 2, c * 4, dropout_rate)
        self.down3 = downsample_block(c * 4, c * 8, dropout_rate)
        self.down4 = downsample_block(c * 8, c * 16, dropout_rate)

        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, kernel_size=3,
                                      stride=2, padding=1, output_padding=1)
        self.up3 = upsample_block(c * 8, c * 4, dropout_rate)
        self.up2 = upsample_block(c * 4, c * 2, dropout_rate)
        self.up1 = upsample_block(c * 2, c, dropout_rate)
        self.up0 = nn.Sequential(nn.Dropout(dropout_rate),
                                 double_conv_block(c * 2, c))

        self.final = nn.Conv2d(c, 1, kernel_size=1)

    def forward(self, x):
        """
        forward pass of UNet with skip concatenation
        :param x: input
        :return: output
        """
        d1 = self.down0(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        u4 = self.up4(d5)
        u3 = self.up3(torch.cat([d4, u4], dim=1))
        u2 = self.up2(torch.cat([d3, u3], dim=1))
        u1 = self.up1(torch.cat([d2, u2], dim=1))
        u0 = self.up0(torch.cat([d1, u1], dim=1))
        return torch.sigmoid(self.final(u0))
