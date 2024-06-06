import torch
from torch import nn
from fas_models.base import ADMMNet, ResNetStack
from fas_models.base.admm_net import ADMMNet_S
from fas_models.base.resnet_ import ResNet18_


class ADMM_ResNet_model(nn.Module):

    def __init__(
            self,
            in_ch=30,
            admm_stages=4,
            admm_channels=(32, 64, 128),
            resnet_repetitions=(2, 2, 2, 2),
            resnet_filters=(32, 64, 128, 256),
            activation=nn.ReLU,
            norm='bn',
            *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = ADMMNet(in_ch, in_ch, admm_stages, admm_channels, activation=activation)
        self.decoder = ResNetStack(in_ch, resnet_repetitions, resnet_filters, activation=activation, norm=norm)

        self.dropout = nn.Dropout()
        self.linear = nn.Linear(resnet_filters[-1], 1)
        self.out_act = nn.Sigmoid()

    def forward(self, inputs):
        x = self.encoder(inputs)
        x = self.decoder(x)

        x = self.dropout(x)
        x = self.linear(x)
        x = self.out_act(x)
        return x


class ADMM_ResNet(nn.Module):

    def __init__(
            self,
            admmnet,
            resnet,
            pretrained_admm_weights=None,
            pretrained_resnet_weights=None,
    ):
        super(ADMM_ResNet, self).__init__()
        self.encoder = admmnet
        self.decoder = resnet
        if pretrained_admm_weights is not None:
            admm_weights = torch.load(pretrained_admm_weights)
            self.encoder.load_state_dict(admm_weights)
        if pretrained_resnet_weights is not None:
            resnet_weights = torch.load(pretrained_resnet_weights)
            self.decoder.load_state_dict(resnet_weights)

    def forward(self, inputs):
        x = self.encoder(inputs)
        x = self.decoder(x)
        return x


class ADMMS_ResNet18(ADMM_ResNet):

    def __init__(
            self,
            pretrained_admm_weights=None,
            pretrained_resnet_weights=None,
            pretrained=False,
    ):
        admmnet = ADMMNet_S(in_ch=30)
        resnet = ResNet18_(in_ch=30, pretrained=pretrained)
        super(ADMMS_ResNet18, self).__init__(
            admmnet=admmnet,
            resnet=resnet,
            pretrained_admm_weights=pretrained_admm_weights,
            pretrained_resnet_weights=pretrained_resnet_weights,
        )


class ADMM_ResNet_S(ADMM_ResNet_model):

    def __init__(
            self,
            in_ch=30,
            admm_stages=2,
            admm_channels=(32, 64, 128),
            resnet_repetitions=(2, 2, 2, 2),
            resnet_filters=(32, 64, 128, 128),
            activation=nn.ReLU,
            norm='bn',
            *args, **kwargs):
        super().__init__(in_ch, admm_stages, admm_channels, resnet_repetitions, resnet_filters, activation, norm, *args, **kwargs)
        

class ResNet_S(nn.Module):
    
    def __init__(
            self,
            repetitions=(2, 2, 2, 2),
            filters=(32, 64, 128, 128),
            activation=nn.ReLU,
            norm='bn',
            *args, **kwargs
    ):
        super(ResNet_S, self).__init__(*args, **kwargs)
        self.resnet = ResNetStack(in_ch=3, repetitions=repetitions, filters=filters, activation=activation, norm=norm)

        self.dropout = nn.Dropout()
        self.linear = nn.Linear(filters[-1], 1)
        self.out_act = nn.Sigmoid()

    def forward(self, inputs):
        x = self.resnet(inputs)

        x = self.dropout(x)
        x = self.linear(x)
        x = self.out_act(x)
        return x


