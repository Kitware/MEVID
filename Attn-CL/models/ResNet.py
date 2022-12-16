from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torchvision

from tools.helplayer import BNClassifier , BottleSoftmax , weights_init_kaiming , weights_init_classifier

__all__ = ['ResNet50TP', 'ResNet50TA' ,  'ResNet50TA_BT' , 'ResNet50TA_BT2']
 
#num classes = 625
#input x = (32, 4, 3, 224, 112)
#person_ids ids =  (32,)
#cam_id c = (32,)

class ResNet50TP(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(ResNet50TP, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.feat_dim = 2048
        self.classifier = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x):
        b = x.size(0)
        t = x.size(1)
        x = x.view(b*t,x.size(2), x.size(3), x.size(4))
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(b,t,-1)
        x=x.permute(0,2,1)
        f = F.avg_pool1d(x,t)
        f = f.view(b, self.feat_dim)
        if not self.training:
            return f
        y = self.classifier(f)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        elif self.loss == {'cent'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


class ResNet50TA(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(ResNet50TA, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.att_gen = 'softmax' # method for attention generation: softmax or sigmoid
        self.feat_dim = 2048 # feature dimension
        self.middle_dim = 256 # middle layer dimension
        self.classifier = nn.Linear(self.feat_dim, num_classes)
        self.attention_conv = nn.Conv2d(self.feat_dim, self.middle_dim, [7,4]) # 7,4 cooresponds to 224, 112 input image size
        self.attention_tconv = nn.Conv1d(self.middle_dim, 1, 3, padding=1)
    def forward(self, x):
        b = x.size(0)
        t = x.size(1)
        x = x.view(b*t, x.size(2), x.size(3), x.size(4))
        x = self.base(x)
        a = F.relu(self.attention_conv(x))
        a = a.view(b, t, self.middle_dim)
        a = a.permute(0,2,1)
        a = F.relu(self.attention_tconv(a))
        a = a.view(b, t)
        x = F.avg_pool2d(x, x.size()[2:])
        if self. att_gen=='softmax':
            a = F.softmax(a, dim=1)
        elif self.att_gen=='sigmoid':
            a = F.sigmoid(a)
            a = F.normalize(a, p=1, dim=1)
        else: 
            raise KeyError("Unsupported attention generation function: {}".format(self.att_gen))
        x = x.view(b, t, -1)
        a = torch.unsqueeze(a, -1)
        a = a.expand(b, t, self.feat_dim)
        att_x = torch.mul(x,a)
        att_x = torch.sum(att_x,1)
        
        f = att_x.view(b,self.feat_dim)
        if not self.training:
            return f
        y = self.classifier(f)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        elif self.loss == {'cent'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))




class ResNet50TA_BT(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(ResNet50TA_BT, self).__init__()
        resnet50 = torchvision.models.resnet50(pretrained=True)
        
        resnet50.layer4[0].conv2.stride = (1,1)
        resnet50.layer4[0].downsample[0].stride = (1,1)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.gap = nn.AdaptiveAvgPool2d(1)


        self.att_gen = 'softmax'
        
        self.feat_dim = 2048 # feature dimension
        self.middle_dim = 256 # middle layer dimension
        # self.classifier = nn.Linear(self.feat_dim, num_classes)
        self.classifier = BNClassifier(self.feat_dim, num_classes , initialization=True)
        
        self.attention_conv = nn.Conv2d(self.feat_dim, self.middle_dim, [14,7]) # 7,4 cooresponds to 224, 112 input image size
        self.attention_tconv = nn.Conv1d(self.middle_dim, 1, 3, padding=1)
        
        self.attention_conv.apply(weights_init_kaiming) 
        self.attention_tconv.apply(weights_init_kaiming) 

        # self.attention_conv.apply(weights_init_classifier)
        # self.attention_tconv.apply(weights_init_classifier)          
        
    def forward(self, x):
        b = x.size(0)
        t = x.size(1)
        x = x.view(b*t, x.size(2), x.size(3), x.size(4))
        x = self.base(x)

        a = F.relu(self.attention_conv(x))
        a = a.view(b, t, self.middle_dim)
        a = a.permute(0,2,1)
        a = F.relu(self.attention_tconv(a))
        a = a.view(b, t)
        x = self.gap(x)
        a = F.softmax(a, dim=1)
        x = x.view(b, t, -1)
        a = torch.unsqueeze(a, -1)
        a = a.expand(b, t, self.feat_dim)
        att_x = torch.mul(x,a)
        att_x = torch.sum(att_x,1)
        
        f = att_x.view(b,self.feat_dim)
        f, y = self.classifier(f)
        if not self.training:
            return f
        return y, f
    



class ResNet50TA_BT2(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(ResNet50TA_BT2, self).__init__()
        resnet50 = torchvision.models.resnet50(pretrained=True)
        
        resnet50.layer4[0].conv2.stride = (1,1)
        resnet50.layer4[0].downsample[0].stride = (1,1)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.gap = nn.AdaptiveAvgPool2d(1)


        self.att_gen = 'softmax'
        
        self.feat_dim = 2048 # feature dimension
        self.middle_dim = 256 # middle layer dimension
        # self.classifier = nn.Linear(self.feat_dim, num_classes)
        self.classifier = BNClassifier(self.feat_dim, num_classes , initialization=True)
        
        self.attention_conv = nn.Conv2d(self.feat_dim, self.middle_dim, [14,7]) # 7,4 cooresponds to 224, 112 input image size
        self.attention_tconv = nn.Conv1d(self.middle_dim, 1, 3, padding=1)
        
        self.attention_conv.apply(weights_init_kaiming) 
        self.attention_tconv.apply(weights_init_kaiming) 

        # self.attention_conv.apply(weights_init_classifier)
        # self.attention_tconv.apply(weights_init_classifier)          
        
    def forward(self, x):
        b = x.size(0)
        t = x.size(1)
        x = x.view(b*t, x.size(2), x.size(3), x.size(4))
        x = self.base(x)

        a = F.relu(self.attention_conv(x))
        a = a.view(b, t, self.middle_dim)
        a = a.permute(0,2,1)
        a = F.relu(self.attention_tconv(a))
        a = a.view(b, t)
        a_vals = a 
        x = self.gap(x)
        a = F.softmax(a, dim=1)
        x = x.view(b, t, -1)
        a = torch.unsqueeze(a, -1)
        a = a.expand(b, t, self.feat_dim)
        att_x = torch.mul(x,a)
        att_x = torch.sum(att_x,1)
        
        f = att_x.view(b,self.feat_dim)
        f, y = self.classifier(f)
        if not self.training:
            return f
        return y, f , a_vals
    
