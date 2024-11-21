import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50, resnet18, resnet34
from torchvision.models.resnet import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights


def spatial_attention(x, mha, norm):
    b, c, h, w = x.size()
    inputs = x  # b, c, h, w
    x = torch.reshape(x, (b, c, h * w))  # b, c, h * w
    x = torch.transpose(x, 1, 2)  # b, h * w, c
    x, _ = mha(x, x, x, need_weights=False)  # b, h * w, c
    x = torch.transpose(x, 1, 2)  # b, c, h * w
    x = torch.reshape(x, (b, c, h, w))  # b, c, h, w
    x = x + inputs
    return norm(x)


class ResNet_(nn.Module):
    def __init__(self, model=resnet50, in_ch=30, num_classes=1, weights=None, attention=False):
        super().__init__()
        self.num_classes = num_classes
        self.model = model(weights=weights)
        self.attention = attention
        if in_ch != 3:
            self.old_conv1 = self.model.conv1
            self.model.conv1 = nn.Conv2d(in_ch, self.old_conv1.out_channels, kernel_size=7, stride=2, padding=3,
                                         bias=False)
            state_d = self.old_conv1.state_dict()
            nmul = in_ch // 3
            if in_ch % 3 != 0:
                nmul = nmul + 1
            state_d['weight'] = torch.tile(state_d['weight'], (1, nmul, 1, 1))[:, :in_ch, :, :]
            self.model.conv1.load_state_dict(state_d)

        self.old_fc = self.model.fc
        self.model.fc = nn.Linear(self.old_fc.in_features, num_classes)

        if self.attention:
            self.mha1 = nn.MultiheadAttention(embed_dim=64, num_heads=2, batch_first=True)
            self.mha2 = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
            self.mha3 = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)
            self.mha4 = nn.MultiheadAttention(embed_dim=512, num_heads=16, batch_first=True)
            self.mha1_norm = nn.BatchNorm2d(num_features=64)
            self.mha2_norm = nn.BatchNorm2d(num_features=128)
            self.mha3_norm = nn.BatchNorm2d(num_features=256)
            self.mha4_norm = nn.BatchNorm2d(num_features=512)
        else:
            self.mha1 = None
            self.mha2 = None
            self.mha3 = None
            self.mha4 = None

    def forward_with_attention(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)  # bsize, 64, 32, 32
        x = spatial_attention(x, self.mha1, self.mha1_norm)

        x = self.model.layer2(x)  # bsize, 128, 16, 16
        x = spatial_attention(x, self.mha2, self.mha2_norm)

        x = self.model.layer3(x)  # bsize, 256, 8, 8
        x = spatial_attention(x, self.mha3, self.mha3_norm)

        x = self.model.layer4(x)  # bsize, 512, 4, 4
        x = spatial_attention(x, self.mha4, self.mha4_norm)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)

        return x

    def forward(self, inputs):
        if not self.attention:
            x = self.model(inputs)
        else:
            x = self.forward_with_attention(inputs)

        if self.num_classes == 1:
            x = torch.sigmoid(x)
        return x


class ResNet18_(ResNet_):
    def __init__(self, in_ch=30, num_classes=1, pretrained=False, attention=False):
        if pretrained:
            weights = ResNet18_Weights.DEFAULT
        else:
            weights = None
        super().__init__(resnet18, in_ch, num_classes, weights, attention)


class ResNet34_(ResNet_):
    def __init__(self, in_ch=30, num_classes=1, pretrained=False, attention=False):
        if pretrained:
            weights = ResNet34_Weights.DEFAULT
        else:
            weights = None
        super().__init__(resnet34, in_ch, num_classes, weights, attention)


class ResNet50_(ResNet_):
    def __init__(self, in_ch=30, num_classes=1, pretrained=False, attention=False):
        if pretrained:
            weights = ResNet50_Weights.DEFAULT
        else:
            weights = None
        super().__init__(resnet50, in_ch, num_classes, weights, attention)


if __name__ == '__main__':
    net_layer = ResNet18_(in_ch=30, num_classes=1, attention=True)

    v = torch.randn((2, 30, 128, 128))
    v = net_layer(v)
    print(v, v.shape)

