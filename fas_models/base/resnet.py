import torch
import torch.nn as nn
from fas_models.base.common import get_norm_layer, ConvNormAct


class BasicBlock(nn.Module):

    def __init__(self, in_ch, filters=128, conv_skip=False, activation=nn.ReLU, norm='bn'):
        super().__init__()

        self.conv_skip = conv_skip
        if conv_skip:
            self.skip_conv = nn.Conv2d(in_ch, filters, 1, 2)
            self.skip_bn = get_norm_layer(filters, norm)
            self.convblock1 = ConvNormAct(in_ch, filters, 3, 2, activation=activation, padding=1, bias=False)
        else:
            self.skip_conv = None
            self.skip_bn = None
            self.convblock1 = ConvNormAct(in_ch, filters, 3, 1, activation=activation, padding=1, bias=False)

        self.conv = nn.Conv2d(filters, filters, 3, 1, padding=1)
        self.bn = get_norm_layer(filters, norm)
        self.act_fcn = activation()

    def forward(self, x):
        if self.conv_skip:
            skip = self.skip_conv(x)
            skip = self.skip_bn(skip)
        else:
            skip = x

        x = self.convblock1(x)
        x = self.conv(x)
        x = self.bn(x)
        x = x + skip
        x = self.act_fcn(x)
        return x


class BottleNeck(nn.Module):

    def __init__(self, in_ch, filters=128, conv_skip=False, activation=nn.ReLU, norm='bn'):
        super().__init__()

        self.conv_skip = conv_skip
        self.convblock1 = ConvNormAct(in_ch, filters, 1, 1, activation=activation, bias=False)
        if conv_skip:
            self.skip_conv = nn.Conv2d(in_ch, filters * 4, 1, 2)
            self.skip_bn = get_norm_layer(filters * 4, norm)
            self.convblock2 = ConvNormAct(filters, filters, 3, 2, padding=1, activation=activation, bias=False)
        else:
            self.convblock2 = ConvNormAct(filters, filters, 3, 1, padding=1, activation=activation, bias=False)
            if in_ch != filters * 4:
                self.skip_conv = nn.Conv2d(in_ch, filters * 4, 1, 1)
                self.skip_bn = get_norm_layer(filters * 4, norm)
            else:
                self.skip_conv = None
                self.skip_bn = None

        self.conv = nn.Conv2d(filters, filters * 4, 1, 1)
        self.bn = get_norm_layer(filters * 4, norm)
        self.act_fcn = activation()

    def forward(self, x):
        if self.skip_conv is not None:
            skip = self.skip_conv(x)
            skip = self.skip_bn(skip)
        else:
            skip = x

        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.conv(x)
        x = self.bn(x)
        x = x + skip
        x = self.act_fcn(x)
        return x


class ResNetSmall(nn.Module):
    def __init__(self, in_ch, repetitions=(2, 2, 2, 2), filters=(64, 128, 256, 512), activation=nn.ReLU, norm='bn',
                 num_classes=1, include_top=False):
        super(ResNetSmall, self).__init__()
        self.conv1 = ConvNormAct(in_ch, filters[0], 7, 2, padding=3, activation=activation, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layers = nn.ModuleList()
        in_ch = filters[0]
        for n, (rep, flt) in enumerate(zip(repetitions, filters)):
            if n == 0:
                self.layers.append(BasicBlock(in_ch, flt, False, activation, norm))
            else:
                self.layers.append(BasicBlock(in_ch, flt, True, activation, norm))
            in_ch = flt
            for _ in range(rep - 1):
                self.layers.append(BasicBlock(in_ch, flt, False, activation, norm))

        self.include_top = include_top
        self.num_classes = num_classes
        self.final_pool = nn.AdaptiveAvgPool2d((1, 1))
        if include_top:
            self.output_fc = nn.Linear(filters[-1], num_classes)
        else:
            self.output_fc = None

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        for layer in self.layers:
            x = layer(x)
        x = self.final_pool(x)
        x = torch.squeeze(x)
        if self.include_top:
            x = self.output_fc(x)
            if self.num_classes == 1:
                x = torch.sigmoid(x)
        return x


class ResNetLarge(nn.Module):

    def __init__(self, in_ch, repetitions=(3, 4, 6, 3), filters=(64, 128, 256, 512), activation=nn.ReLU, norm='bn',
                 num_classes=1, include_top=False):
        super(ResNetLarge, self).__init__()
        self.conv1 = ConvNormAct(in_ch, filters[0], 7, 2, padding=3, activation=activation, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layers = nn.ModuleList()
        in_ch = filters[0]
        for n, (rep, flt) in enumerate(zip(repetitions, filters)):
            if n == 0:
                self.layers.append(BottleNeck(in_ch, flt, False, activation, norm))
            else:
                self.layers.append(BottleNeck(in_ch, flt, True, activation, norm))
            in_ch = flt * 4
            for _ in range(rep - 1):
                self.layers.append(BottleNeck(in_ch, flt, False, activation, norm))

        self.include_top = include_top
        self.num_classes = num_classes
        self.final_pool = nn.AdaptiveAvgPool2d((1, 1))
        if include_top:
            self.output_fc = nn.Linear(filters[-1] * 4, num_classes)
        else:
            self.output_fc = None

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        for layer in self.layers:
            x = layer(x)
        x = self.final_pool(x)
        x = torch.squeeze(x)
        if self.include_top:
            x = self.output_fc(x)
            if self.num_classes == 1:
                x = torch.sigmoid(x)
        return x


class ResNet18(ResNetSmall):

    def __init__(self, in_ch=30, num_classes=1, include_top=True):
        super(ResNet18, self).__init__(
            in_ch=in_ch,
            repetitions=(2, 2, 2, 2),
            num_classes=num_classes,
            include_top=include_top,
        )


class ResNet34(ResNetSmall):

    def __init__(self, in_ch=30, num_classes=1, include_top=True):
        super(ResNet34, self).__init__(
            in_ch=in_ch,
            repetitions=(3, 4, 6, 3),
            num_classes=num_classes,
            include_top=include_top,
        )


class ResNet50(ResNetLarge):

    def __init__(self, in_ch=30, num_classes=1, include_top=True):
        super(ResNet50, self).__init__(
            in_ch=in_ch,
            repetitions=(3, 4, 6, 3),
            num_classes=num_classes,
            include_top=include_top,
        )


class ResNet101(ResNetLarge):

    def __init__(self, in_ch=30, num_classes=1, include_top=True):
        super(ResNet101, self).__init__(
            in_ch=in_ch,
            repetitions=(3, 4, 23, 3),
            num_classes=num_classes,
            include_top=include_top,
        )


class ResNetStack(nn.Module):

    def __init__(self, in_ch, repetitions=(2, 2, 2, 2), filters=(32, 64, 128, 256), activation=nn.ReLU, norm='bn',
                 num_classes=1, include_top=False):
        super().__init__()
        self.layers = nn.ModuleList()
        for rep, flt in zip(repetitions, filters):
            self.layers.append(BottleNeck(in_ch, flt, conv_skip=True, activation=activation, norm=norm))
            in_ch = flt
            for _ in range(rep - 1):
                self.layers.append(BottleNeck(in_ch, flt, conv_skip=False, activation=activation, norm=norm))
        self.final_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.include_top = include_top
        self.num_classes = num_classes
        if include_top:
            self.output_fc = nn.Linear(filters[-1], num_classes)
        else:
            self.output_fc = None

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.final_pool(x)
        x = torch.squeeze(x)
        if self.include_top:
            x = self.output_fc(x)
            if self.num_classes == 1:
                x = torch.sigmoid(x)
        return x