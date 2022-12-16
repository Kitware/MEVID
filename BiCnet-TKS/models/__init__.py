from __future__ import absolute_import

from .BiCnet_TKS import *



__factory = {
    'BiCnet_TKS': BiCnet_TKS,
}

def get_names():
    return __factory.keys()

def init_model(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))
    return __factory[name](*args, **kwargs)
