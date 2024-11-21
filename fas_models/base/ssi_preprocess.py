import torch
from torch import nn


class EnergyNormalization(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = torch.nn.Parameter(torch.Tensor([0]))

    def forward(self, y, phi, phi_s):
        equalized = torch.div(y, phi_s + self.gamma)
        equalized = torch.unsqueeze(equalized, dim=1)
        x = equalized * phi
        x = torch.concat([x, equalized], dim=1)
        return x


class SensingMatrixNormalization(nn.Module):

    def __init__(self, in_ch=30, norm='ln', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = torch.nn.Parameter(torch.Tensor([0]))
        if norm == 'ln':
            self.norm = nn.GroupNorm(1, in_ch)
        elif norm == 'bn':
            self.norm = nn.BatchNorm2d(in_ch)
        else:
            raise ValueError()
        self.epsilon = 1e-6

    def forward(self, y, phi, phi_s):
        y = torch.unsqueeze(y, dim=1)
        equalized = torch.div(phi, y + self.gamma + self.epsilon)
        equalized = self.norm(equalized)
        return equalized


class SimpleConcat(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, y, phi, phi_s):
        b, h, w = y.shape

        maxy = torch.max(torch.reshape(y, (b, -1)), dim=1, keepdim=True)[0]
        maxy = torch.reshape(maxy, (b, 1, 1))
        y = y / maxy
        y = torch.unsqueeze(y, dim=1)

        maxp = torch.max(torch.reshape(phi, (b, -1)), dim=1, keepdim=True)[0]
        maxp = torch.reshape(maxp, (b, 1, 1, 1))
        phi = phi / maxp
        x = torch.concat([y, phi], dim=1)
        return x * 2. - 1.

