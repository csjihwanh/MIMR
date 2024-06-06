# Code Modified from original source: https://github.com/milesial/Pytorch-UNet/blob/67bf11b4db4c5f2891bd7e8e7f58bcde8ee2d2db/unet/unet_model.py
""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 4)
        self.down1 = Down(4, 8)
        self.down2 = Down(8, 8)
        self.down3 = Down(8, 16)
        factor = 2 if bilinear else 1
        self.down4 = Down(16, 32 // factor)
        self.up1 = Up(32, 16 // factor, bilinear)
        self.up2 = Up(16, 16 // factor, bilinear)
        self.up3 = Up(16, 8 // factor, bilinear)
        self.up4 = Up(8, 4, bilinear)
        self.outc = OutConv(4, n_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.sigmoid(self.outc(x))
        return logits