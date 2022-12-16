import torch
import torch.nn as nn
import torch.nn.functional as F

if __package__:
    from .resnet3d import *
else:
    from resnet3d import *

__all__ = ['ResNet3dT', 'resnet3dt50']

class ResNet3dT(nn.Module):
    networks = {'resnet3d10': None,
                'resnet3d18': None,
                'resnet3d34': None,
                'resnet3d50': './pretrained/resnet-50-kinetics.pth'}
    def __init__(self, network, num_classes, loss={'xent', 'htri'}, pretrained='', **kwargs):
        super(ResNet3dT, self).__init__(**kwargs)
        assert network in self.networks, '{} is not supported'.format(network)
        if not pretrained:
            pretrained = self.networks[network]

        resnet3d = eval(network)(pretrained=pretrained)
        self.base = nn.Sequential(*list(resnet3d.children())[:-1])
        self.num_classes = num_classes
        self.loss = loss
        self.fc = nn.Linear(resnet3d.fc.in_features, num_classes)

    def forward(self, x):
        # default size is (b, s, c, w, h), s for seq_len, c for channel
        # convert for 3d cnn, (b, c, s, w, h)
        x = x.permute(0, 2, 1, 3, 4)
        x = self.base(x)
        x = F.adaptive_avg_pool3d(x, (1, 1, 1))
        x = x.view(x.size(0), -1)
        if not self.training:
            return x

        y = self.fc(x)
        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, x
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

def resnet3dt50(**kwargs):
    return ResNet3dT(network='resnet3d50', **kwargs)

if __name__ == '__main__':
    model = resnet3dt50(num_classes=625, pretrained='../../pretrained/resnet-50-kinetics.pth')
    model(torch.randn((1, 4, 3, 224, 112)))