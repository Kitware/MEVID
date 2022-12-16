from __future__ import absolute_import

import os
import shutil
import inspect

# video
from .res50tp import *
from .sta import *
from .simple_sta import *
from .gsta import *
from .resnet50_s1 import *
from .graphnet import *
from .vmgn import *
from .ganet import *

__model_factory = {
    'res50tp': res50tp,
    'resnet50_s1': resnet50_s1,
    'sta': sta_p4,
    'simple_sta': simple_sta_p4,
    'gsta': gsta,
    'msppn': MSPyraPartNet,
    'msppgn': MSPyraPartGraphNet,
    'vmgn': vmgn,
    'ganet': ganet
}


def get_names():
    return list(__model_factory.keys())


def init_model(name, *args, **kwargs):
    if name not in list(__model_factory.keys()):
        raise KeyError("Unknown model: {}".format(name))
    if 'save_dir' in kwargs:
        # XXX: shutil.copy and shutil.copy2 raise PermissionError, so use copyfile here.
        model_file = inspect.getfile(__model_factory[name])
        shutil.copyfile(model_file, os.path.join(os.path.abspath(kwargs['save_dir']), os.path.basename(model_file)))
    return __model_factory[name](*args, **kwargs)
