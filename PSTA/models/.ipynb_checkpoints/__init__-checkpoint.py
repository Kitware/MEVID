from __future__ import absolute_import

from models.app_refine import app_refine
from models.baseline import ResNet50
from models.app_tem import app_tem
from models.temporal_dense import tem_dense

__factory = {
    'app_refine' : app_refine,
    'ResNet50' : ResNet50,
    'app_tem': app_tem,
    'tem_dense' : tem_dense
}

def get_names():
    return __factory.keys()

def init_model(name,*args,**kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown model :{}".format(name))
    return __factory[name](*args,**kwargs)