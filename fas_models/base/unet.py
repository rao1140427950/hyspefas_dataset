import torch
import torch.nn as nn
from fas_models.base.common import ConvNormAct, ConvTranspose2dNormAct


class DownsampleBlock(nn.Module):

    def __init__(self, in_ch, nfilters, nconvs, activation=nn.ReLU, bias=True):
        super(DownsampleBlock, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(nconvs):
            self.layers.append(ConvNormAct(in_ch, nfilters, kernel_size=3, strides=1, padding=1, activation=activation,
                                           bias=bias))
            in_ch = nfilters
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        skip = x
        x = self.maxpool(x)
        return x, skip


class UpsampleBlock(nn.Module):

    def __init__(self, in_ch, nfilters, nconvs, activation=nn.ReLU, bias=True):
        super(UpsampleBlock, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(nconvs):
            self.layers.append(ConvNormAct(in_ch, nfilters, kernel_size=3, strides=1, padding=1, activation=activation,
                                           bias=bias))
            in_ch = nfilters
        self.convt = ConvTranspose2dNormAct(in_ch, in_ch // 2, kernel_size=2, strides=2, padding=0,
                                               activation=activation, bias=bias)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.convt(x)
        return x


class UNet(nn.Module):

    def __init__(self, in_ch=30, out_ch=1, base_ffdim=32, ndownsample=3, nconvs_per_stage=3, activation=nn.ReLU, bias=True):
        super(UNet, self).__init__()
        self.out_ch = out_ch
        self.down_layers = nn.ModuleList()
        ffdim = base_ffdim
        for n in range(ndownsample):
            self.down_layers.append(
                DownsampleBlock(in_ch, ffdim, nconvs_per_stage, activation, bias)
            )
            in_ch = ffdim
            ffdim *= 2

        # 16*16*128
        self.up_layers = nn.ModuleList()
        for n in range(ndownsample):
            self.up_layers.append(
                UpsampleBlock(in_ch, ffdim, nconvs_per_stage, activation, bias)
            )
            in_ch = ffdim
            ffdim = ffdim // 2

        self.out_conv = nn.Sequential(
            ConvNormAct(in_ch, ffdim, kernel_size=3, strides=1, padding=1, activation=activation, bias=bias),
            ConvNormAct(ffdim, ffdim, kernel_size=3, strides=1, padding=1, activation=activation, bias=bias),
            ConvNormAct(ffdim, ffdim, kernel_size=3, strides=1, padding=1, activation=activation, bias=bias),
            nn.Conv2d(ffdim, out_ch, 1, 1, bias=bias),
        )

    def forward(self, x):
        skips = []
        for layer in self.down_layers:
            x, skip = layer(x)
            skips.append(skip)

        skips = skips[::-1]
        for n, layer in enumerate(self.up_layers):
            x = layer(x)
            x = torch.concat([x, skips[n]], dim=1)

        x = self.out_conv(x)
        if self.out_ch == 1:
            x = torch.squeeze(x)
            x = torch.sigmoid(x)
        return x


class UNet_S(UNet):

    def __init__(
            self,
            in_ch=30,
            out_ch=1,
            activation=nn.ReLU,
            bias=True,
    ):
        super(UNet_S, self).__init__(
            in_ch=in_ch,
            out_ch=out_ch,
            base_ffdim=16,
            ndownsample=3,
            nconvs_per_stage=3,
            activation=activation,
            bias=bias,
        )


if __name__ == '__main__':
    unet = UNet()
    v = torch.randn((2, 30, 128, 128))
    y = unet(v)
    print(y)
