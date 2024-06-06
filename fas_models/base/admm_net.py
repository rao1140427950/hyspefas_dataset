"""
credit to:
    https://github.com/mengziyi64/ADMM-net/tree/master
    https://github.com/caiyuanhao1998/MST/tree/main
"""

import torch
import torch.nn as nn
from fas_models.base.common import DoubleConv


def A(x, Phi):
    temp = x * Phi
    y = torch.sum(temp, 1)
    return y

def At(y, Phi):
    temp = torch.unsqueeze(y, 1).repeat(1, Phi.shape[1], 1, 1)
    x = temp * Phi
    return x


class SimpleUNet(nn.Module):

    def __init__(self, in_ch, out_ch=30, channels=(32, 64, 128), activation=nn.ReLU, *args, **kwargs):
        super(SimpleUNet, self).__init__(*args, **kwargs)

        self.dconv_down1 = DoubleConv(in_ch, channels[0], activation=activation)
        self.dconv_down2 = DoubleConv(channels[0], channels[1], activation=activation)
        self.dconv_down3 = DoubleConv(channels[1], channels[2], activation=activation)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(channels[2], channels[1], kernel_size=2, stride=2),
            activation(),
        )
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(channels[1], channels[0], kernel_size=2, stride=2),
            activation(),
        )
        self.dconv_up2 = DoubleConv(channels[1] * 2, channels[1], activation=activation)
        self.dconv_up1 = DoubleConv(channels[0] * 2, channels[0], activation=activation)

        self.conv_last = nn.Conv2d(channels[0], out_ch, 1)
        self.afn_last = nn.Tanh()

    def forward(self, x):
        inputs = x
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)

        x = self.upsample2(conv3)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample1(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        x = self.conv_last(x)
        x = self.afn_last(x)
        out = x + inputs

        return out


class ADMMLayer(nn.Module):

    def __init__(self, in_ch, out_cn=30, channels=(32, 64, 128), activation=nn.ReLU, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unet = SimpleUNet(in_ch, out_cn, channels, activation)
        self.gamma = torch.nn.Parameter(torch.Tensor([0]))

    def forward(self, y, Phi, Phi_s, theta, b):
        yb = A(theta + b, Phi)
        x = theta + b + At(torch.div(y - yb, Phi_s + self.gamma), Phi)
        x1 = x - b
        theta = self.unet(x1)
        b = b - (x - theta)

        return theta, b


class ADMMNet(nn.Module):

    def __init__(self, in_ch, out_ch=30, num_stages=9, channels=(32, 64, 128), activation=nn.ReLU, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = nn.ModuleList()
        for _ in range(num_stages):
            self.layers.append(ADMMLayer(in_ch, out_ch, channels, activation))

    def forward(self, inputs):
        y, Phi, Phi_s = inputs
        theta = At(y, Phi)
        b = torch.zeros_like(Phi)

        for layer in self.layers:
            theta, b = layer(y, Phi, Phi_s, theta, b)

        return theta


class ADMMNet_S(ADMMNet):

    def __init__(
            self,
            in_ch=30,
            out_ch=30,
    ):
        super(ADMMNet_S, self).__init__(
            in_ch=in_ch,
            out_ch=out_ch,
            num_stages=2,
            channels=(32, 64, 128),
        )


class ADMMNet_M(ADMMNet):

    def __init__(
            self,
            in_ch=30,
            out_ch=30,
    ):
        super(ADMMNet_M, self).__init__(
            in_ch=in_ch,
            out_ch=out_ch,
            num_stages=6,
            channels=(32, 64, 128),
        )


if __name__ == '__main__':
    net_layer = ADMMNet(num_stages=4, in_ch=30, out_ch=30)

    vy = torch.randn((2, 256, 256))
    vPhi = torch.randn((30, 256, 256))
    vPhi_s = torch.sum(vPhi, dim=0)
    vPhi = torch.unsqueeze(vPhi, 0).repeat((2, 1, 1, 1))
    vPhi_s = torch.unsqueeze(vPhi_s, 0).repeat((2, 1, 1))
    v = net_layer(vy, vPhi, vPhi_s)
    print(v, v.shape)