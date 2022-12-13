import torch
from torch import nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F

# from models.refine_method import *
from models.backbone.resnet import *

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
}

def init_pretrained_weight(model,model_url):
    """Initializes model with pretrained weight

    Layers that don't match with pretrained layers in name or size are kept unchanged
    """
    pretrain_dict = model_zoo.load_url(model_url, model_dir = './')
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k,v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)

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

def weight_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight,std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias,0.0)


class app_refine(nn.Module):

    def __init__(self, num_classes,model_name, pretrain_choice,seq_len):
        super(app_refine,self).__init__()
        if model_name == 'resnet50':
            self.in_planes = 2048
            self.base = ResNet(last_stride=1,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])

        if pretrain_choice == 'imagenet':
            init_pretrained_weight(self.base,model_urls[model_name])
            print('Loading pretrained ImageNet model......')

        self.num_classes = num_classes
        self.part_num = 4
        self.feat_pool = nn.AdaptiveAvgPool3d((1,1,1))
        self.local_part_avgpool = nn.AdaptiveAvgPool2d((self.part_num,1))

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.classifier = nn.Linear(self.in_planes,self.num_classes,bias=False)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weight_init_classifier)

    def global_Center_cosine(self,feat_vec):
        b, t, _ = feat_vec.size()

        similar_matrix = torch.zeros(b, t, t)

        for i in range(t):
            for j in range(t):
                similar_matrix[:, i, j] = torch.cosine_similarity(feat_vec[:, i, :], feat_vec[:, j, :])

        similar_score = torch.sum(similar_matrix, 2, keepdim=True)
        remove_id = torch.argmin(similar_score, 1)
        refine_feature = torch.zeros(b, t - 1, feat_vec.size(2))

        for i in range(b):
            refine_feature[i] = feat_vec[i, torch.arange(t) != remove_id[i], :]  # b*t-1*10

        #L2 normalize
        refine_feature = refine_feature.cuda()  #(32,3,2048)
        norm_score = torch.norm(refine_feature, 2, dim=2).unsqueeze(2)
        refine_feature = refine_feature / norm_score
        #L1 normalize
        #norm_score = torch.norm(refine_feature, 1, dim=2).unsqueeze(2)
        #refine_feature = refine_feature / norm_score

        #或者把度量的距离改成欧式距离
        # cosine_sum_similar = 0
        # for i in range(t - 1):
        #     for j in range(i + 1, t - 1):
        #         cosine_similar_score = torch.cosine_similarity(refine_feature[:, i, :], refine_feature[:, j, :])
                #归一化处理：cosine_similar_score = torch.div(cosine_similar_score + 1, 2)
                # cosine_similar_score = torch.sigmoid(cosine_similar_score)
                # cosine_similar_score = -torch.log(cosine_similar_score)
                # cosine_sum_similar = cosine_sum_similar + cosine_similar_score

        return refine_feature

    def global_KNN_cosine(self,feat_vec):  # 这个要要验证一下，cosine_similarity的输出是不是可以跨维度的。   r()eply:cosine_similarity是可以跨维度，或者说可以保持维度
        b, t, _ = feat_vec.size()
        refine_feature = torch.zeros(b, t - 1, feat_vec.size(2))
        feat_vec_avg = torch.mean(feat_vec, 1)
        similar_matrix = torch.zeros(b, t)

        for i in range(t):
            similar_score = torch.cosine_similarity(feat_vec_avg, feat_vec[:, i, :])
            similar_matrix[:, i] = similar_score

        remove_id = torch.argmin(similar_matrix, 1)

        for i in range(b):
            refine_feature[i] = feat_vec[i, torch.arange(t) != remove_id[i], :]  # b*t-1*1024

        refine_feature = refine_feature.cuda()  # (32,3,2048)
        norm_score = torch.norm(refine_feature, 2, dim=2).unsqueeze(2)
        refine_feature = refine_feature / norm_score

        # cosine_sum_similar = 0
        # for i in range(t - 1):
        #     for j in range(i + 1, t - 1):
        #         cosine_similar_score = torch.cosine_similarity(refine_feature[:, i, :], refine_feature[:, j, :])
        #         cosine_similar_score = torch.sigmoid(cosine_similar_score)         # 成比例压缩到(0,1)区间
        #         cosine_similar_score = - torch.log(cosine_similar_score)
        #         cosine_sum_similar = cosine_sum_similar + cosine_similar_score

        return refine_feature

    def local_Center_cosine(self, feat_vec):
        b, t, c, n = feat_vec.size()
        similar_matrix = torch.zeros(b, n, t, t)

        for i in range(n):
            for j in range(t):
                for k in range(t):
                    similar_matrix[:, i, j, k] = torch.cosine_similarity(feat_vec[:, j, :, i], feat_vec[:, k, :, i])

        similar_matrix = torch.sum(similar_matrix, 2)
        remove_id = torch.argmax(similar_matrix, 2)
        refine_local_feature = torch.zeros(b, t - 1, c, n)

        for i in range(b):
            for j in range(n):
                refine_local_feature[i, :, :, j] = feat_vec[i, torch.arange(t) != remove_id[i,j], :, j]

        refine_local_feature = refine_local_feature.permute(0,1,3,2)
        refine_local_feature = torch.mean(refine_local_feature, 2).cuda()

        norm_score = torch.norm(refine_local_feature, 2, dim = 2).unsqueeze(2)
        refine_local_feature = refine_local_feature / norm_score

        return refine_local_feature

    def forward(self,x, pids = None, camids = None):
        b,t,c,w,h = x.size()
        x = x.view(b*t,c,w,h)
        feat_map = self.base(x)      #(b*t,c,16,8)
#         print(feat_map)

#         feat_local_vec = self.local_part_avgpool(feat_map)
#         feat_local_vec = feat_local_vec.view(b, t, -1, self.part_num)

#         if self.local_refine_method == 'Center_cosine':
#             refine_local_feature = self.local_Center_cosine(feat_local_vec)
#         elif self.local_refine_method == 'KNN_cosine':
#             pass

#         feat_global_vec = self.feat_pool(feat_map)
#         feat_global_vec = feat_global_vec.view(b, t, -1)  #(b,t,1024)

#         if self.global_refine_method == 'KNN_cosine':
#             refine_global_feature = self.global_KNN_cosine(feat_global_vec)
#         # elif self.global_refine_method == 'KNN_dist':
#         #     refine_global_feature, appearance_param = global_KNN_dist(feat_global_vec)
#         elif self.global_refine_method == 'Center_cosine':
#             refine_global_feature = self.global_Center_cosine(feat_global_vec)
#         # elif self.global_refine_method == 'Center_dist':
#         #     refine_global_feature, appearance_param = global_Center_dist(feat_global_vec)

#         refine_feature = 0.5 * refine_local_feature + 0.5 * refine_global_feature

#         cosine_sum_similar = 0
#         for i in range(t - 1):
#             for j in range(i + 1, t - 1):
#                 cosine_similar_score = torch.cosine_similarity(refine_feature[:, i, :], refine_feature[:, j, :])
#                 # 归一化处理：cosine_similar_score = torch.div(cosine_similar_score + 1, 2)
#                 cosine_similar_score = torch.sigmoid(cosine_similar_score)
#                 cosine_similar_score = -torch.log(cosine_similar_score)
#                 cosine_sum_similar = cosine_sum_similar + cosine_similar_score


#         credible_feature = torch.mean(refine_global_feature,1)
#         appearance_loss = torch.mean(cosine_sum_similar).cuda()
        
        _, c, w, h = feat_map.size()
        feat_map = feat_map.view(b, t, c, w, h).permute(0, 2, 1, 3, 4)
        feature = self.feat_pool(feat_map).view(b,-1)
        BN_feature = self.bottleneck(feature)

        if self.training:
            cls_score = self.classifier(BN_feature)
            return cls_score, feature
        else:
            return BN_feature, pids, camids


# if __name__ == '__main__':
#     x = torch.rand(16,5,3,256,128)
#     num_classes = 625
#     last_stride = 1
#     model_path = '/home/wyq/.torch/models/resnet50-19c8e357.pth'
#     neck = 'no'
#     model_name = 'resnet50'
#     pretrain_choice = 'imagenet'
#     model = app_refine(num_classes,last_stride,model_path,neck,model_name,pretrain_choice)
#     model(x)