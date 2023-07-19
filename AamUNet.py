from UNet import *


def upsample_block_(in_channels, out_channels, dropout):
    """
    an upsampling block in the expansive path of a UNet
    :param in_channels: int, usually multiple of two
    :param out_channels: int, usually half as in_channels
    :param dropout: dropout
    :return: the upsampling block
    """
    return nn.Sequential(
        nn.Dropout(dropout),
        double_conv_block(in_channels, in_channels),
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2,
                           padding=1, output_padding=1))


class AAM(nn.Module):
    """
    This augmented attention module takes in an upsampled high-level feature map
    of shape (bs, high_ch, H, W) and a low-level spatial map of shape (bs,
    low_ch, H, W) and outputs, through applying attention, a map of shape (bs,
    high_ch, H, W)
    """
    def __init__(self, low_ch, high_ch):
        super(AAM, self).__init__()
        self.global_pooling = nn.AdaptiveAvgPool2d(1)

        self.conv1 = nn.Sequential(
            nn.Conv2d(low_ch, high_ch, 1, padding=0),
            nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(
            nn.Conv2d(high_ch, high_ch, 1, padding=0),
            nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(
            nn.Conv2d(high_ch, high_ch, 1, padding=0),
            nn.Softmax(dim=1))

        self.conv4 = nn.Sequential(
            nn.Conv2d(low_ch, high_ch, 1, padding=0),
            nn.BatchNorm2d(high_ch),
            nn.ReLU(inplace=True))

    def forward(self, input_high, input_low):
        mid_high = self.global_pooling(input_high)
        weight_high = self.conv1(mid_high)

        mid_low = self.global_pooling(input_low)
        weight_low = self.conv2(mid_low)

        weight = self.conv3(weight_low + weight_high)
        low = self.conv4(input_low)
        return input_high + low.mul(weight)


class AamUNet(nn.Module):
    """
    Augmented Attention Module UNet
    """

    def __init__(self, cfg):
        """
        initializer of UNet
        :param cfg: the model configurations
        """
        super(AamUNet, self).__init__()
        c = cfg["channel"]
        dropout_rate = cfg["dropout"]

        self.down0 = double_conv_block(3, c)
        self.down1 = downsample_block(c, c * 2, dropout_rate)
        self.down2 = downsample_block(c * 2, c * 4, dropout_rate)
        self.down3 = downsample_block(c * 4, c * 8, dropout_rate)
        self.down4 = downsample_block(c * 8, c * 16, dropout_rate)

        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, kernel_size=3,
                                      stride=2, padding=1, output_padding=1)
        self.aam3 = AAM(c * 8, c * 8)
        self.up3 = upsample_block_(c * 8, c * 4, dropout_rate)
        self.aam2 = AAM(c * 4, c * 4)
        self.up2 = upsample_block_(c * 4, c * 2, dropout_rate)
        self.aam1 = AAM(c * 2, c * 2)
        self.up1 = upsample_block_(c * 2, c, dropout_rate)
        self.aam0 = AAM(c, c)
        self.up0 = nn.Sequential(nn.Dropout(dropout_rate),
                                 double_conv_block(c, c))

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
        u3 = self.up3(self.aam3(u4, d4))
        u2 = self.up2(self.aam2(u3, d3))
        u1 = self.up1(self.aam1(u2, d2))
        u0 = self.up0(self.aam0(u1, d1))
        return torch.sigmoid(self.final(u0))
