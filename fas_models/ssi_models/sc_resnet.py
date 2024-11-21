# import torch
from torch import nn
from fas_models.base import ResNetStack, SimpleConcat
from fas_models.base.resnet_ import ResNet18_, ResNet34_


class SC_ResNet_model(nn.Module):

    def __init__(
            self,
            in_ch=30,
            resnet_repetitions=(2, 2, 2, 2),
            resnet_filters=(32, 64, 128, 256),
            activation=nn.ReLU,
            norm='bn',
            *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = SimpleConcat()
        self.decoder = ResNetStack(in_ch + 1, resnet_repetitions, resnet_filters, activation=activation, norm=norm)

        self.dropout = nn.Dropout()
        self.linear = nn.Linear(resnet_filters[-1], 1)
        self.out_act = nn.Sigmoid()

    def forward(self, inputs):
        meas, phi, phi_s = inputs
        x = self.encoder(meas, phi, phi_s)
        x = self.decoder(x)

        x = self.dropout(x)
        x = self.linear(x)
        x = self.out_act(x)
        return x


class SC_ResNet_S(SC_ResNet_model):

    def __init__(
            self,
            in_ch=30,
            resnet_repetitions=(2, 2, 2, 2),
            resnet_filters=(32, 64, 128, 128),
            activation=nn.ReLU,
            norm='bn',
            *args, **kwargs):
        super().__init__(in_ch, resnet_repetitions, resnet_filters, activation, norm, *args, **kwargs)


class SC_ResNet_model_(nn.Module):

    def __init__(
            self,
            in_ch=30,
            resnet_=None,
            pretrained=False,
    ):
        super().__init__()
        self.encoder = SimpleConcat(in_ch)
        self.decoder = resnet_(in_ch + 1, pretrained=pretrained)


    def forward(self, inputs):
        meas, phi, phi_s = inputs
        x = self.encoder(meas, phi, phi_s)
        x = self.decoder(x)

        return x


class SC_ResNet18_(SC_ResNet_model_):

    def __init__(
            self,
            in_ch=30,
            pretrained=False,
    ):
        super().__init__(in_ch, ResNet18_, pretrained)


class SC_ResNet34_(SC_ResNet_model_):

    def __init__(
            self,
            in_ch=30,
            pretrained=False,
    ):
        super().__init__(in_ch, ResNet34_, pretrained)