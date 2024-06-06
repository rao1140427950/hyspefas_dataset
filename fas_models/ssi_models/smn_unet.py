import random

import torch
from torch import nn
from fas_models.base import SensingMatrixNormalization
from fas_models.base.common import SpecDecoder
from fas_models.base.unet import UNet


class SMN_UNet_model(nn.Module):

    def __init__(
            self,
            in_ch=30,
            num_classes=1,
            base_ffdim=32,
            ndownsample=3,
            nconvs_per_stage=3,
            ):
        super().__init__()
        self.encoder = SensingMatrixNormalization(in_ch)
        self.decoder = UNet(in_ch=in_ch, out_ch=num_classes, base_ffdim=base_ffdim, ndownsample=ndownsample,
                            nconvs_per_stage=nconvs_per_stage)

    def forward(self, inputs):
        meas, phi, phi_s = inputs
        x = self.encoder(meas, phi, phi_s)
        x = self.decoder(x)

        return x


class SMN_Unet_S(SMN_UNet_model):

    def __init__(
            self,
            in_ch=30,
            num_classes=1,
    ):
        super(SMN_Unet_S, self).__init__(
            in_ch=in_ch,
            num_classes=num_classes,
            base_ffdim=16,
            ndownsample=3,
            nconvs_per_stage=3,
        )


class SMN_Unet_M(SMN_UNet_model):

    def __init__(
            self,
            in_ch=30,
            num_classes=30,
    ):
        super(SMN_Unet_M, self).__init__(
            in_ch=in_ch,
            num_classes=num_classes,
            base_ffdim=16,
            ndownsample=4,
            nconvs_per_stage=4,
        )


class SMN_UNet_model_V2(nn.Module):

    def __init__(
            self,
            in_ch=30,
            num_classes=1,
            base_ffdim=16,
            ndownsample=4,
            nconvs_per_stage=4,
            ):
        super().__init__()
        self.encoder = SensingMatrixNormalization(in_ch)
        self.spec_decoder = SpecDecoder(in_ch, nblocks=3, ksize=5, in_ffdim=256)
        self.unet = UNet(in_ch=1, out_ch=base_ffdim, base_ffdim=base_ffdim, ndownsample=ndownsample,
                            nconvs_per_stage=nconvs_per_stage)
        self.out_conv = nn.Conv2d(base_ffdim, 1, kernel_size=3, padding=1)

    def forward(self, inputs):
        meas, phi, phi_s = inputs
        x = self.encoder(meas, phi, phi_s)
        rec1 = self.spec_decoder(x)
        n, c, h, w = rec1.shape

        x = torch.reshape(rec1, (n * c, h, w))
        x = self.unet(x)
        x = self.out_conv(x)
        rec2 = torch.reshape(x, (n, c, h, w))

        if self.training:
            if random.randint(0, 1) == 0:
                out = rec1
            else:
                out = rec2
        else:
            out = rec2

        return out


class SMN_Unet_M_V2(SMN_UNet_model):

    def __init__(
            self,
            in_ch=30,
            num_classes=30,
    ):
        super(SMN_Unet_M_V2, self).__init__(
            in_ch=in_ch,
            num_classes=num_classes,
            base_ffdim=32,
            ndownsample=3,
            nconvs_per_stage=3,
        )