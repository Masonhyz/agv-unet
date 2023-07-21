from AgUNet import *
from ResUNet import *


class AgResUNet(ResUNet):
    def __init__(self, configurations):
        super(AgResUNet, self).__init__(configurations)
        c = configurations["root channel"]
        # b = configurations["batch normalization"]
        # d = configurations["dropout"]

        # self.down1 = DownsampleResBlock(3, c, b, d)
        # self.down2 = DownsampleResBlock(c, c * 2, b, d)
        # self.down3 = DownsampleResBlock(c * 2, c * 4, b, d)
        # self.down4 = DownsampleResBlock(c * 4, c * 8, b, d)
        # self.bottleneck = DoubleConvolutionBlock1(c * 8, c * 16, b)
        # self.btn_res_conv = nn.Conv2d(c*8, c*16, 1)
        # self.btn_relu = nn.ReLU(inplace=True)
        self.ag4 = AG([c*16, c*8], c*8)
        # self.up4 = UpsampleResBlock(c * 16, c * 8, b, d)
        self.ag3 = AG([c*8, c*4], c*4)
        # self.up3 = UpsampleResBlock(c * 8, c * 4, b, d)
        self.ag2 = AG([c*4, c*2], c*2)
        # self.up2 = UpsampleResBlock(c * 4, c * 2, b, d)
        self.ag1 = AG([c*2, c*1], c*1)
        # self.up1 = UpsampleResBlock(c * 2, c, b, d)
        # self.final = nn.Conv2d(c, 1, 1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1, y1 = self.down1(x)
        x2, y2 = self.down2(y1)
        x3, y3 = self.down3(y2)
        x4, y4 = self.down4(y3)
        btn_res = self.btn_res_conv(y4)
        btn = self.bottleneck(y4)
        btn = btn_res + btn
        btn = self.btn_relu(btn)
        u4 = self.up4(self.ag4(btn, x4), btn)
        u3 = self.up3(self.ag3(u4, x3), u4)
        u2 = self.up2(self.ag2(u3, x2), u3)
        u1 = self.up1(self.ag1(u2, x1), u2)
        out = self.sigmoid(self.final(u1))
        return out
