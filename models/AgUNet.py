from models.UNet import *


class AG(nn.Module):
    """
    This attention gate module takes in a high level feature information of
    shape (bs, in_channels[0], H, W) and a low level spatial information of
    shape (bs, in_channels[1], H*2, W*2) and outputs, through applying
    attention, an upsampled map of shape (bs, in_channels[1], H*2, W*2).

    Precondition: in_channels[0]==2*in_channels[1];
    and int_channels==in_channels[1], this also does matter.
    """
    def __init__(self, in_channels, int_channels):
        super(AG, self).__init__()
        self.wg = nn.Conv2d(in_channels[0], int_channels, kernel_size=1)
        self.wx = nn.Conv2d(in_channels[1], int_channels, kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.psi = nn.Conv2d(int_channels, 1, kernel_size=1)
        self.sig = nn.Sigmoid()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, g, x):
        wg = self.wg(g)
        wx = self.wx(x)
        a = wg + wx
        a = self.relu(a)
        a = self.psi(a)
        a = self.sig(a)
        a = self.up(a)
        out = a * x
        # could add a batch normalization here but since the x was already batch
        # normalized during double convolution block in the encoder path, it
        # would be redundant
        return out


class AgUNet(UNet):
    """
    Attention gated UNet, a subclass of UNet.
    """
    def __init__(self, configurations):
        """
        Fully configurable initializer, with a fixed number of depth: 5
        :param configurations: dict, an attribute of a Configs instance.
        """
        super(AgUNet, self).__init__(configurations)
        c = configurations["root channel"]
        # b = configurations["batch normalization"]
        # d = configurations["dropout"]

        # self.down1 = DownsampleBlock(3, c, b, d)
        # self.down2 = DownsampleBlock(c, c*2, b, d)
        # self.down3 = DownsampleBlock(c*2, c*4, b, d)
        # self.down4 = DownsampleBlock(c*4, c*8, b, d)
        # self.bottleneck = DoubleConvolutionBlock(c*8, c*16, b)
        self.ag4 = AG([c*16, c*8], c*8)
        # self.up4 = UpsampleBlock(c*16, c*8, b, d)
        self.ag3 = AG([c*8, c*4], c*4)
        # self.up3 = UpsampleBlock(c*8, c*4, b, d)
        self.ag2 = AG([c*4, c*2], c*2)
        # self.up2 = UpsampleBlock(c*4, c*2, b, d)
        self.ag1 = AG([c*2, c*1], c*1)
        # self.up1 = UpsampleBlock(c*2, c, b, d)
        # self.final = nn.Conv2d(c, 1, 1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        AgUNet forward function
        :param x: input
        :return: output
        """
        x1, y1 = self.down1(x)
        x2, y2 = self.down2(y1)
        x3, y3 = self.down3(y2)
        x4, y4 = self.down4(y3)
        btn = self.bottleneck(y4)
        u4 = self.up4(self.ag4(btn, x4), btn)
        u3 = self.up3(self.ag3(u4, x3), u4)
        u2 = self.up2(self.ag2(u3, x2), u3)
        u1 = self.up1(self.ag1(u2, x1), u2)
        out = self.sigmoid(self.final(u1))
        return out
