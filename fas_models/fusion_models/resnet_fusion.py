import torch
import torch.nn as nn
from fas_models.base.resnet_ import ResNet18_, spatial_attention
from fas_models.base.ssi_preprocess import SensingMatrixNormalization


class ResNet_Fusion_(nn.Module):

    def __init__(self, ssi_in_ch, rgb_in_ch, resnet=None, num_classes=1, main_course='ssi', pretrained=False):
        super().__init__()

        self.ssi_resnet = resnet(in_ch=ssi_in_ch, num_classes=num_classes, pretrained=pretrained).model
        self.rgb_resnet = resnet(in_ch=rgb_in_ch, num_classes=num_classes, pretrained=pretrained).model

        self.main_course = main_course
        self.num_classes = num_classes

    def forward(self, inputs):
        ssi, rgb = inputs

        ssi = self.ssi_resnet.conv1(ssi)
        ssi = self.ssi_resnet.bn1(ssi)
        ssi = self.ssi_resnet.relu(ssi)
        ssi = self.ssi_resnet.maxpool(ssi)

        rgb = self.rgb_resnet.conv1(rgb)
        rgb = self.rgb_resnet.bn1(rgb)
        rgb = self.rgb_resnet.relu(rgb)
        rgb = self.rgb_resnet.maxpool(rgb)

        if self.main_course == 'ssi':
            mnet = self.ssi_resnet
            anet = self.rgb_resnet
            m = ssi
            a = rgb
        elif self.main_course == 'rgb':
            mnet = self.rgb_resnet
            anet = self.ssi_resnet
            m = rgb
            a = ssi
        else:
            raise ValueError()

        m = mnet.layer1(m)
        a = anet.layer1(a)
        m = m + a

        m = mnet.layer2(m)
        a = anet.layer2(a)
        m = m + a

        m = mnet.layer3(m)
        a = anet.layer3(a)
        m = m + a

        m = mnet.layer4(m)
        a = anet.layer4(a)
        m = m + a

        m = mnet.avgpool(m)
        m = torch.flatten(m, 1)
        m = mnet.fc(m)

        if self.num_classes == 1:
            m = torch.sigmoid(m)

        return m


class ResNet18_Fusion_(ResNet_Fusion_):

    def __init__(self, ssi_in_ch=30, rgb_in_ch=3, num_classes=1, main_course='ssi', pretrained=False):
        super().__init__(
            ssi_in_ch=ssi_in_ch,
            rgb_in_ch=rgb_in_ch,
            resnet=ResNet18_,
            num_classes=num_classes,
            main_course=main_course,
            pretrained=pretrained,
        )


class SMN_ResNet_Fusion_(nn.Module):

    def __init__(self, ssi_in_ch=30, rgb_in_ch=3, resnet=None, num_classes=1, main_course='ssi', pretrained=False):
        super(SMN_ResNet_Fusion_, self).__init__()
        self.ssi_preprocess = SensingMatrixNormalization(ssi_in_ch)
        self.main_network = ResNet_Fusion_(ssi_in_ch=ssi_in_ch, rgb_in_ch=rgb_in_ch, resnet=resnet,
                                           num_classes=num_classes, main_course=main_course, pretrained=pretrained)

    def forward(self, inputs):
        meas, phi, phi_s, rgb = inputs
        ssi = self.ssi_preprocess(meas, phi, phi_s)
        return self.main_network((ssi, rgb))


class SMN_ResNet18_Fusion_(SMN_ResNet_Fusion_):

    def __init__(
            self,
            ssi_in_ch=30,
            rgb_in_ch=3,
            num_classes=1,
            main_course='ssi',
            pretrained=False,
    ):
        super(SMN_ResNet18_Fusion_, self).__init__(
            ssi_in_ch=ssi_in_ch,
            rgb_in_ch=rgb_in_ch,
            resnet=ResNet18_,
            num_classes=num_classes,
            main_course=main_course,
            pretrained=pretrained,
        )


def spatial_cross_attention(q, kv, mha, norm):
    b, c, h, w = kv.size()
    inputs = x = kv  # b, c, h, w
    x = torch.reshape(x, (b, c, h * w))  # b, c, h * w
    x = torch.transpose(x, 1, 2)  # b, h * w, c
    q = torch.reshape(q, (b, c, h * w))  # b, c, h * w
    q = torch.transpose(q, 1, 2)  # b, h * w, c

    x, _ = mha(q, x, x, need_weights=False)  # b, h * w, c

    x = torch.transpose(x, 1, 2)  # b, c, h * w
    x = torch.reshape(x, (b, c, h, w))  # b, c, h, w
    x = x + inputs
    return norm(x)


def spatial_cross_attention_v2(qk, v, mha, norm):
    b, c, h, w = v.size()
    inputs = x = v  # b, c, h, w
    x = torch.reshape(x, (b, c, h * w))  # b, c, h * w
    x = torch.transpose(x, 1, 2)  # b, h * w, c
    qk = torch.reshape(qk, (b, c, h * w))  # b, c, h * w
    qk = torch.transpose(qk, 1, 2)  # b, h * w, c

    x, _ = mha(qk, qk, x, need_weights=False)  # b, h * w, c

    x = torch.transpose(x, 1, 2)  # b, c, h * w
    x = torch.reshape(x, (b, c, h, w))  # b, c, h, w
    x = x + inputs
    return norm(x)


class ResNet_AttentionFusion_(nn.Module):

    def __init__(self, ssi_in_ch, rgb_in_ch, resneta=None, num_classes=1, pretrained=False):
        super().__init__()

        self.ssi_resneta = resneta(in_ch=ssi_in_ch, num_classes=num_classes, pretrained=pretrained, attention=True)
        self.rgb_resneta = resneta(in_ch=rgb_in_ch, num_classes=num_classes, pretrained=pretrained, attention=True)

        self.mha1 = nn.MultiheadAttention(embed_dim=64, num_heads=2, batch_first=True)
        self.mha2 = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        self.mha3 = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)
        self.mha4 = nn.MultiheadAttention(embed_dim=512, num_heads=16, batch_first=True)
        self.mha1_norm = nn.BatchNorm2d(num_features=64)
        self.mha2_norm = nn.BatchNorm2d(num_features=128)
        self.mha3_norm = nn.BatchNorm2d(num_features=256)
        self.mha4_norm = nn.BatchNorm2d(num_features=512)

        self.num_classes = num_classes


    def forward(self, inputs):
        ssi, rgb = inputs

        ssi = self.ssi_resneta.model.conv1(ssi)
        ssi = self.ssi_resneta.model.bn1(ssi)
        ssi = self.ssi_resneta.model.relu(ssi)
        ssi = self.ssi_resneta.model.maxpool(ssi)

        rgb = self.rgb_resneta.model.conv1(rgb)
        rgb = self.rgb_resneta.model.bn1(rgb)
        rgb = self.rgb_resneta.model.relu(rgb)
        rgb = self.rgb_resneta.model.maxpool(rgb)

        ssi = self.ssi_resneta.model.layer1(ssi)
        ssi = spatial_attention(ssi, self.ssi_resneta.mha1, self.ssi_resneta.mha1_norm)
        rgb = self.rgb_resneta.model.layer1(rgb)
        rgb = spatial_attention(rgb, self.rgb_resneta.mha1, self.rgb_resneta.mha1_norm)
        ssi = spatial_cross_attention(rgb, ssi, self.mha1, self.mha1_norm)

        ssi = self.ssi_resneta.model.layer2(ssi)
        ssi = spatial_attention(ssi, self.ssi_resneta.mha2, self.ssi_resneta.mha2_norm)
        rgb = self.rgb_resneta.model.layer2(rgb)
        rgb = spatial_attention(rgb, self.rgb_resneta.mha2, self.rgb_resneta.mha2_norm)
        ssi = spatial_cross_attention(rgb, ssi, self.mha2, self.mha2_norm)

        ssi = self.ssi_resneta.model.layer3(ssi)
        ssi = spatial_attention(ssi, self.ssi_resneta.mha3, self.ssi_resneta.mha3_norm)
        rgb = self.rgb_resneta.model.layer3(rgb)
        rgb = spatial_attention(rgb, self.rgb_resneta.mha3, self.rgb_resneta.mha3_norm)
        ssi = spatial_cross_attention(rgb, ssi, self.mha3, self.mha3_norm)

        ssi = self.ssi_resneta.model.layer4(ssi)
        ssi = spatial_attention(ssi, self.ssi_resneta.mha4, self.ssi_resneta.mha4_norm)
        rgb = self.rgb_resneta.model.layer4(rgb)
        rgb = spatial_attention(rgb, self.rgb_resneta.mha4, self.rgb_resneta.mha4_norm)
        ssi = spatial_cross_attention(rgb, ssi, self.mha4, self.mha4_norm)

        ssi = self.ssi_resneta.model.avgpool(ssi)
        ssi = torch.flatten(ssi, 1)
        ssi = self.ssi_resneta.model.fc(ssi)

        if self.num_classes == 1:
            ssi = torch.sigmoid(ssi)

        return ssi


class ResNet18_AttentionFusion_(ResNet_AttentionFusion_):

    def __init__(self, ssi_in_ch=30, rgb_in_ch=3, num_classes=1, pretrained=False):
        super().__init__(
            ssi_in_ch=ssi_in_ch,
            rgb_in_ch=rgb_in_ch,
            resneta=ResNet18_,
            num_classes=num_classes,
            pretrained=pretrained,
        )


class SMN_ResNet_AttentionFusion_(nn.Module):

    def __init__(self, ssi_in_ch=30, rgb_in_ch=3, resneta=None, num_classes=1, pretrained=False):
        super(SMN_ResNet_AttentionFusion_, self).__init__()
        self.ssi_preprocess = SensingMatrixNormalization(ssi_in_ch)
        self.main_network = ResNet_AttentionFusion_(ssi_in_ch=ssi_in_ch, rgb_in_ch=rgb_in_ch, resneta=resneta,
                                           num_classes=num_classes, pretrained=pretrained)

    def forward(self, inputs):
        meas, phi, phi_s, rgb = inputs
        ssi = self.ssi_preprocess(meas, phi, phi_s)
        return self.main_network((ssi, rgb))


class SMN_ResNet18_AttentionFusion_(SMN_ResNet_AttentionFusion_):

    def __init__(
            self,
            ssi_in_ch=30,
            rgb_in_ch=3,
            num_classes=1,
            pretrained=False,
    ):
        super(SMN_ResNet18_AttentionFusion_, self).__init__(
            ssi_in_ch=ssi_in_ch,
            rgb_in_ch=rgb_in_ch,
            resneta=ResNet18_,
            num_classes=num_classes,
            pretrained=pretrained,
        )


if __name__ == '__main__':
    x1 = torch.randn((2, 128, 128))
    mask = torch.randn((2, 30, 128, 128))
    mask_s = torch.randn((2, 128, 128))
    x2 = torch.randn((2, 3, 128, 128))

    network = SMN_ResNet18_AttentionFusion_()
    yy = network((x1, mask, mask_s, x2))