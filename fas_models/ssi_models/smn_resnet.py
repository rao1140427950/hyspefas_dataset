# import torch
from torch import nn
from fas_models.base import SensingMatrixNormalization
from fas_models.base.resnet_ import ResNet18_, ResNet34_, ResNet50_


class SMN_ResNet_model_(nn.Module):

    def __init__(
            self,
            in_ch=30,
            resnet_=None,
            pretrained=False,
            **kwargs,
    ):
        super().__init__()
        self.encoder = SensingMatrixNormalization(in_ch)
        self.decoder = resnet_(in_ch, pretrained=pretrained, **kwargs)
        self.model = self.decoder.model


    def forward(self, inputs):
        meas, phi, phi_s = inputs
        x = self.encoder(meas, phi, phi_s)
        x = self.decoder(x)

        return x


class SMN_ResNet18_(SMN_ResNet_model_):

    def __init__(
            self,
            in_ch=30,
            pretrained=False,
            **kwargs,
    ):
        super().__init__(in_ch, ResNet18_, pretrained, **kwargs)


class SMN_ResNet34_(SMN_ResNet_model_):

    def __init__(
            self,
            in_ch=30,
            pretrained=False,
            **kwargs,
    ):
        super().__init__(in_ch, ResNet34_, pretrained, **kwargs)


class SMN_ResNet50_(SMN_ResNet_model_):

    def __init__(
            self,
            in_ch=30,
            pretrained=False,
            **kwargs,
    ):
        super().__init__(in_ch, ResNet50_, pretrained, **kwargs)