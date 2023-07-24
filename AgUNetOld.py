from UNetOld import *


class AG(nn.Module):
    """
    This attention gate module takes in a high level feature information of
    shape (bs, in_channels[0], H, W) and a low level spatial information of
    shape (bs, in_channels[1], H*2, W*2) and outputs, through applying
    attention, an upsampled map of shape (bs, in_channels[1], H*2, W*2).

    Precondition: in_channels[0]==2*in_channels[1];
    and usually int_channels==in_channels[1] but this does not really matter.
    """
    def __init__(self, in_channels, int_channels):
        super(AG, self).__init__()
        self.wg = nn.Conv2d(in_channels[0], int_channels, kernel_size=1)

        self.wx = nn.Conv2d(in_channels[1], int_channels, kernel_size=2, stride=2)

        self.relu = nn.ReLU(inplace=True)
        self.psi = nn.Conv2d(int_channels, 1, kernel_size=1)
        self.sig = nn.Sigmoid()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.bn = nn.BatchNorm2d(in_channels[1])

    def forward(self, g, x):
        wg = self.wg(g)
        wx = self.wx(x)
        a = wg + wx
        a = self.relu(a)
        a = self.psi(a)
        a = self.sig(a)
        a = self.up(a)
        out = a * x
        out = self.bn(out)
        return out


class AgUNet(nn.Module):
    """
    Classic Attention UNet class with Attention Gates
    """

    def __init__(self, cfg):
        """
        initializer of UNet
        :param cfg: the models configurations
        """
        super(AgUNet, self).__init__()
        c = cfg["root channel"]
        dropout_rate = cfg["dropout"]

        self.down0 = double_conv_block(3, c)
        self.down1 = downsample_block(c, c*2, dropout_rate)
        self.down2 = downsample_block(c*2, c*4, dropout_rate)
        self.down3 = downsample_block(c*4, c*8, dropout_rate)
        self.down4 = downsample_block(c*8, c*16, dropout_rate)

        self.ag4 = AG([c * 16, c * 8], c * 8)
        self.up4 = nn.ConvTranspose2d(c*16, c*8, kernel_size=3,
                                      stride=2, padding=1, output_padding=1)
        self.ag3 = AG([c * 8, c * 4], c * 4)
        self.up3 = Upsample_block(c*8, c*4, dropout_rate)
        self.ag2 = AG([c * 4, c * 2], c * 2)
        self.up2 = Upsample_block(c*4, c*2, dropout_rate)
        self.ag1 = AG([c * 2, c * 1], c * 1)
        self.up1 = Upsample_block(c*2, c, dropout_rate)
        self.up0 = nn.Sequential(nn.Dropout(dropout_rate),
                                 double_conv_block(c*2, c))

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
        g4 = self.down4(d4)
        u4 = self.up4(g4)
        g3, u3 = self.up3(torch.cat([self.ag4(g4, d4), u4], dim=1))
        g2, u2 = self.up2(torch.cat([self.ag3(g3, d3), u3], dim=1))
        g1, u1 = self.up1(torch.cat([self.ag2(g2, d2), u2], dim=1))
        u0 = self.up0(torch.cat([self.ag1(g1, d1), u1], dim=1))
        return torch.sigmoid(self.final(u0))
