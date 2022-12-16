from __future__ import absolute_import

from .ResNet import *

__factory = {
    'resnet50tp': ResNet50TP,
    'resnet50ta': ResNet50TA,
    'ResNet50ta_bt': ResNet50TA_BT,
    'ResNet50ta_bt2': ResNet50TA_BT2,
}

def get_names():
    return __factory.keys()

def init_model(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))
    return __factory[name](*args, **kwargs)
