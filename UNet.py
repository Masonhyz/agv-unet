import torch
import torch.nn as nn
from tqdm import tqdm
from utils import iou


class DoubleConvolutionBlock(nn.Module):
    def __init__(self, in_channel, out_channel, batch_norm):
        super(DoubleConvolutionBlock, self).__init__()
        self.bn = batch_norm
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, padding=1)
        if batch_norm:
            self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, padding=1)
        if batch_norm:
            self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        if self.bn:
            x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        if self.bn:
            x = self.bn2(x)
        x = self.relu2(x)
        return x


class DownsampleBlock(nn.Module):
    def __init__(self, in_channel, out_channel, batch_norm, dropout):
        super(DownsampleBlock, self).__init__()
        self.double_conv = DoubleConvolutionBlock(in_channel, out_channel, batch_norm)
        self.maxpool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, x):
        x = self.double_conv(x)
        y = self.maxpool(x)
        y = self.dropout(y)
        return x, y


class UpsampleBlock(nn.Module):
    def __init__(self, in_channel, out_channel, batch_norm, dropout):
        super(UpsampleBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channel, in_channel // 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.double_conv = DoubleConvolutionBlock(in_channel, out_channel, batch_norm)

    def forward(self, x, g):
        g = self.up(g)
        y = torch.cat([x, g], dim=1)
        y = self.dropout(y)
        y = self.double_conv(y)
        return y


class UNet(nn.Module):
    def __init__(self, configurations):
        super(UNet, self).__init__()

        c = configurations["root channel"]
        b = configurations["batch normalization"]
        d = configurations["dropout"]

        self.down1 = DownsampleBlock(3, c, b, d)
        self.down2 = DownsampleBlock(c, c*2, b, d)
        self.down3 = DownsampleBlock(c*2, c*4, b, d)
        self.down4 = DownsampleBlock(c*4, c*8, b, d)
        self.bottleneck = DoubleConvolutionBlock(c*8, c*16, b)
        self.up4 = UpsampleBlock(c*16, c*8, b, d)
        self.up3 = UpsampleBlock(c*8, c*4, b, d)
        self.up2 = UpsampleBlock(c*4, c*2, b, d)
        self.up1 = UpsampleBlock(c*2, c, b, d)
        self.final = nn.Conv2d(c, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1, y1 = self.down1(x)
        x2, y2 = self.down2(y1)
        x3, y3 = self.down3(y2)
        x4, y4 = self.down4(y3)
        btn = self.bottleneck(y4)
        u4 = self.up4(x4, btn)
        u3 = self.up3(x3, u4)
        u2 = self.up2(x2, u3)
        u1 = self.up1(x1, u2)
        out = self.sigmoid(self.final(u1))
        return out

    def evaluate(self, data_loader, criterion=iou):
        total_accuracy = 0
        total_num = 0
        for __, images, targets in tqdm(data_loader, desc="Eval", leave=False):
            p = self.forward(images)
            predictions = (p > 0.5).float()
            batch_accuracy = criterion(predictions.detach(), targets.detach())
            total_accuracy += batch_accuracy * images.shape[0]
            total_num += images.shape[0]
        return total_accuracy / total_num
