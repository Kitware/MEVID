import torch
import torch.nn as nn
from .backbones.resnet import ResNet, Bottleneck
import copy
from .backbones.vit_pytorch import vit_base_patch16_224_PiT, vit_small_patch16_224_PiT, deit_small_patch16_224_PiT
from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss
import pdb
def shuffle_unit(features, shift, group, begin=1):
    B, N, M, D = features.shape
    batchsize = B*N
    dim = features.size(-1)
    features = features.contiguous().reshape(B*N,M,D)
    # Shift Operation
    feature_random = torch.cat([features[:, begin-1+shift:], features[:, begin:begin-1+shift]], dim=1)
    x = feature_random
    # Patch Shuffle Operation
    try:
        x = x.view(batchsize, group, -1, dim)
    except:
        x = torch.cat([x, x[:, -2:-1, :]], dim=1)
        x = x.view(batchsize, group, -1, dim)

    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, dim).reshape(B,N,M-1,D)

    return x

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

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Backbone(nn.Module):
    def __init__(self, num_classes, cfg):
        super(Backbone, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT

        if model_name == 'resnet50':
            self.in_planes = 2048
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
            print('using resnet50 as a backbone')
        else:
            print('unsupported backbone! but got {}'.format(model_name))

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None):  # label is unused if self.cos_layer == 'no'
        x = self.base(x)
        global_feat = nn.functional.avg_pool2d(x, x.shape[2:4])
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)

        if self.training:
            if self.cos_layer:
                cls_score = self.arcface(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cls_score, global_feat
        else:
            if self.neck_feat == 'after':
                return feat
            else:
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory):
        super(build_transformer, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768
        self.isVideo = True if cfg.DATASETS.NAMES in ['mars', 'duke-video-reid', 'ilids', 'prid'] else False

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0
        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
                                                        camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH,
                                                        drop_rate= cfg.MODEL.DROP_OUT,
                                                        attn_drop_rate=cfg.MODEL.ATT_DROP_RATE,
                                                        isVideo=self.isVideo)
        if cfg.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224_PiT':
            self.in_planes = 384
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None, cam_label= None, view_label=None):
        global_feat = self.base(x, cam_label=cam_label, view_label=view_label)

        feat = self.bottleneck(global_feat)

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)

            return cls_score, global_feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class build_transformer_local(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory, num,rearrange):
        super(build_transformer_local, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768
        self.isVideo = True if cfg.DATASETS.NAMES in ['mars', 'duke-video-reid', 'ilids', 'prid'] else False
        self.spatial = cfg.MODEL.SPATIAL
        self.temporal = cfg.MODEL.TEMPORAL
        self.diversity = cfg.MODEL.DIVERSITY
        self.vis = cfg.TEST.VIS

        self.pyramid_type = [cfg.MODEL.PYRAMID0_TYPE, cfg.MODEL.PYRAMID1_TYPE,cfg.MODEL.PYRAMID2_TYPE,
                             cfg.MODEL.PYRAMID3_TYPE,cfg.MODEL.PYRAMID4_TYPE]
        self.layer_division_type = [cfg.MODEL.LAYER0_DIVISION_TYPE, cfg.MODEL.LAYER1_DIVISION_TYPE,
                                    cfg.MODEL.LAYER2_DIVISION_TYPE, cfg.MODEL.LAYER3_DIVISION_TYPE,
                                    cfg.MODEL.LAYER4_DIVISION_TYPE]
        self.layer_combination = cfg.MODEL.LAYER_COMBIN
        self.layer_division_num = [num[i] for i in self.layer_division_type]

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0

        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
                                                        local_feature=cfg.MODEL.JPM, camera=camera_num, view=view_num,
                                                        stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH,
                                                        isVideo=self.isVideo, spatial=self.spatial, temporal=self.temporal,
                                                        vis=self.vis)

        if pretrain_choice == 'imagenet':
            if self.isVideo:
                self.base.load_spatiotemporal_param(model_path)
            else:
                self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        block = self.base.head_block
        layer_norm = self.base.norm

        self.pyramid_layer = nn.ModuleList()
        for i in range(self.layer_combination):
            self.pyramid_layer.append(nn.Sequential(copy.deepcopy(block), copy.deepcopy(layer_norm)))

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.ModuleList()
            self.bottleneck = nn.ModuleList()
            for i in range(self.layer_combination):
                self.classifier.append(nn.ModuleList())
                self.bottleneck.append(nn.ModuleList())
                for j in range(self.layer_division_num[i]):
                    self.classifier[i].append(nn.Linear(self.in_planes, self.num_classes, bias=False))
                    self.classifier[i][j].apply(weights_init_classifier)
                    self.bottleneck[i].append(nn.BatchNorm1d(self.in_planes))
                    self.bottleneck[i][j].bias.requires_grad_(False)
                    self.bottleneck[i][j].apply(weights_init_kaiming)

        self.shuffle_groups = cfg.MODEL.SHUFFLE_GROUP
        print('using shuffle_groups size:{}'.format(self.shuffle_groups))
        self.shift_num = cfg.MODEL.SHIFT_NUM
        print('using shift_num size:{}'.format(self.shift_num))
        self.divide_length = cfg.MODEL.DEVIDE_LENGTH
        print('using divide_length size:{}'.format(self.divide_length))
        self.rearrange = rearrange

    def forward(self, x, label=None, cam_label= None, view_label=None):  # label is unused if self.cos_layer == 'no'
        B, N, C, H, W = x.shape

        if self.vis:
            features, attn_base = self.base(x, cam_label=cam_label, view_label=view_label)
        else:
            features = self.base(x, cam_label=cam_label, view_label=view_label)

        feat = [None] * self.layer_combination
        cls_score = [None] * self.layer_combination
        diversity_loss = None
        if self.diversity:
            diversity_loss = [None] * self.layer_combination
        for i in range(self.layer_combination):
            feat[i] = [None] * self.layer_division_num[i]
            cls_score[i] = [None] * self.layer_division_num[i]
            if self.diversity:
                diversity_loss[i] = [None] * self.layer_division_num[i]
            token = features[:, :, 0:1]
            feature = features[:, :, 1:].reshape(B,N,self.base.patch_embed.num_y, self.base.patch_embed.num_x, -1)

            # different division strategy
            if self.pyramid_type[i] == 'patch':
                if self.layer_division_num[i] == 6:
                    local_feats = [feature[:, :, m * 7:(m + 1) * 7, n * 5:(n + 1) * 5, :].reshape(B, N, 35, -1)
                                  for m in range(3) for n in range(2)]
                elif self.layer_division_num[i] == 14:
                    local_feats = [feature[:, :, m * 3:(m + 1) * 3, n * 5:(n + 1) * 5, :].reshape(B, N, 15, -1)
                                  for m in range(7) for n in range(2)]
                elif self.layer_division_num[i] == 15:
                    local_feats = [feature[:, :, m * 7:(m + 1) * 7, n * 2:(n + 1) * 2, :].reshape(B, N, 14, -1)
                                  for m in range(3) for n in range(5)]
            else:
                if self.pyramid_type[i] == 'horizontal':
                    feature = feature.reshape(B, N, -1, self.in_planes)
                elif self.pyramid_type[i] == 'vertical':
                    feature = feature.transpose(-3,-2).reshape(B, N, -1, self.in_planes)
                division_length = (features.size(2) - 1) // self.layer_division_num[i]
                local_feats = [feature[:, :, m * division_length:(m + 1) * division_length]
                               for m in range(self.layer_division_num[i])]

            # deal with each stipe/patch
            attns = []
            for j in range(self.layer_division_num[i]):
                local_feat = torch.cat((token, local_feats[j]), dim=2)
                if self.diversity:
                    local_feat, diver_loss = self.pyramid_layer[i][0](local_feat, return_attention=True)
                    local_feat = self.pyramid_layer[i][1](local_feat)
                    diversity_loss[i][j] = sum(diver_loss) / len(diver_loss)
                elif self.vis:
                    local_feat, attn = self.pyramid_layer[i][0](local_feat, return_attention=True)
                    local_feat = self.pyramid_layer[i][1](local_feat)
                    attns.append(attn)
                else:
                    local_feat = self.pyramid_layer[i](local_feat)
                feat[i][j] = local_feat[:, :, 0].mean(dim=1)
                # if self.training:
                #     feat[i][j] = local_feat[:, :, 0].mean(dim=1)
                # else:
                #     feat[i][j] = local_feat[:, :, 0].mean(dim=1) / self.layer_division_num[i]
                local_feat_bn = self.bottleneck[i][j](feat[i][j])
                cls_score[i][j] = self.classifier[i][j](local_feat_bn)

            if self.vis:
                # different combinations according to division strategy
                return [attn_base, attns]

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                return cls_score, feat, diversity_loss # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                return torch.cat(
                    [feat, local_feat_1_bn / 5, local_feat_2_bn / 5, local_feat_3_bn / 5, local_feat_4_bn / 5, local_feat_5_bn / 5], dim=1)
            else:
                return torch.cat([j for i in feat for j in i], dim=1)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


__factory_T_type = {
    'vit_base_patch16_224_PiT': vit_base_patch16_224_PiT,
    'deit_base_patch16_224_PiT': vit_base_patch16_224_PiT,
    'vit_small_patch16_224_PiT': vit_small_patch16_224_PiT,
    'deit_small_patch16_224_PiT': deit_small_patch16_224_PiT
}

__num_of_layers = {
    '1x210' : 1,    # global
    '2x105' : 2,    # horizontal
    '3x70'  : 3,
    '5x42'  : 5,
    '6x35'  : 6,
    '7x10'  : 7,
    '105x2' : 2,    # vertical
    '70x3'  : 3,
    '42x5'  : 5,
    '35x6'  : 6,
    '10x7'  : 7,
    '6p'    : 6,    # patch
    '14p'   : 14,
    '15p'   : 15,
    'NULL'  : 0,
}

def make_model(cfg, num_class, camera_num, view_num):
    if cfg.MODEL.NAME == 'transformer':
        if cfg.MODEL.JPM:
            model = build_transformer_local(num_class, camera_num, view_num, cfg, __factory_T_type,
                                            __num_of_layers, rearrange=cfg.MODEL.RE_ARRANGE)
            print('===========building transformer with JPM module ===========')
        else:
            model = build_transformer(num_class, camera_num, view_num, cfg, __factory_T_type)
            print('===========building transformer===========')
    else:
        model = Backbone(num_class, cfg)
        print('===========building ResNet===========')
    return model
