import torch
import torch.nn as nn
import torchvision



def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('LayerNorm') != -1:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('GroupNorm') != -1:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class BNClassifier(nn.Module):

    def __init__(self, in_dim, class_num, initialization=True):
        super(BNClassifier, self).__init__()

        self.in_dim = in_dim
        self.class_num = class_num

        self.bn = nn.BatchNorm1d(self.in_dim)
        self.bn.bias.requires_grad_(False)
        self.classifier = nn.Linear(self.in_dim, self.class_num, bias=False)
        if initialization :
            self.bn.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        feature = self.bn(x)
        cls_score = self.classifier(feature)
        return feature, cls_score




class Bottle(nn.Module):
        def forward(self, input):
            if len(input.size()) <= 2:
                return super(Bottle, self).forward(input)
            size = input.size()[:2]
            out = super(Bottle, self).forward(input.view(size[0]*size[1], -1))
            return out.contiguous().view(size[0], size[1], -1)


class BottleSoftmax(Bottle, nn.Softmax):
    pass



class LayerNorm2(nn.Module):
    # features : channels , dim1 : height , dim2: width
    def __init__(self, eps=1e-6):
        super(LayerNorm2, self).__init__()
        self.eps = eps

    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        mean = mean.expand(x.size(0) , x.size(1) , x.size(2) , x.size(3) )
        std = x.std(1, keepdim=True)
        std = std.expand(x.size(0) , x.size(1) , x.size(2) , x.size(3) )
        z = (x - mean) / (std + self.eps)
        return z




class LayerNorm1(nn.Module):
    # features : channels , 
    def __init__(self, eps=1e-6 , features=2048):
        super(LayerNorm1, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        mean = mean.expand(x.size(0) , x.size(1))
        std = x.std(1, keepdim=True)
        std = std.expand(x.size(0) , x.size(1) )
        z = (x - mean) / (std + self.eps)
        bias = self.bias.unsqueeze(0)
        bias = bias.expand(x.size(0) , x.size(1))
        weight = self.weight.unsqueeze(0)
        weight = weight.expand(x.size(0) , x.size(1))
        return weight * z + bias


