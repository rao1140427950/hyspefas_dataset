from torch import nn


def get_norm_layer(num_channels, type_='bn'):
    if type_ == 'bn':
        return nn.BatchNorm2d(num_channels)
    elif type_ == 'gn':
        return nn.GroupNorm(1, num_channels)
    elif type_ == 'ln':
        return nn.LayerNorm(num_channels)
    else:
        raise ValueError('Unknown norm type: `{:s}`'.format(type_))


class ConvNormAct(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=3, strides=1, padding=0, activation=nn.ReLU, norm='bn', bias=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, strides, padding, bias=bias),
            get_norm_layer(out_ch, norm),
            activation(),
        )

    def forward(self, x):
        return self.conv(x)


class ConvTranspose2dNormAct(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=2, strides=2, padding=0, activation=nn.ReLU, norm='bn', bias=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size, strides, padding, bias=bias),
            get_norm_layer(out_ch, norm),
            activation(),
        )

    def forward(self, x):
        return self.conv(x)


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, activation=nn.ReLU):
        super(DoubleConv, self).__init__()
        self.d_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            activation(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            activation(),
        )

    def forward(self, x):
        x = self.d_conv(x)
        return x


class DenseConvResBlock(nn.Module):

    def __init__(
            self,
            in_ch=30,
            r=1.,
    ):
        super(DenseConvResBlock, self).__init__()
        self.conv1 = ConvNormAct(in_ch, in_ch, kernel_size=1)
        mid_ch = round(in_ch * r)
        self.conv2 = ConvNormAct(in_ch, mid_ch, kernel_size=1)
        self.conv3 = ConvNormAct(mid_ch, in_ch, kernel_size=1)

    def forward(self, x):
        skip = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return skip + x


class SpecDecoder(nn.Module):

    def __init__(
            self,
            in_ch=30,
            in_ffdim=128,
            out_ch=30,
            nblocks=3,
            ksize=5,
    ):
        super(SpecDecoder, self).__init__()
        self.in_conv = ConvNormAct(in_ch, in_ffdim, kernel_size=ksize, strides=1, padding=2)
        self.stage1 = nn.ModuleList()
        self.stage2 = nn.ModuleList()
        for n in range(nblocks):
            self.stage1.append(DenseConvResBlock(in_ffdim, r=2))
            self.stage2.append(DenseConvResBlock(in_ffdim, r=0.5))
        self.out_conv = nn.Conv2d(in_ffdim, out_channels=out_ch, kernel_size=1)

    def forward(self, x):
        x = self.in_conv(x)
        for layer in self.stage1:
            x = layer(x)
        for layer in self.stage2:
            x = layer(x)
        x = self.out_conv(x)
        return x
