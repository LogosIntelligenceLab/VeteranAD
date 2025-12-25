# Copyright (c) OpenMMLab. All rights reserved.
import importlib
import os
import pkgutil
import warnings
from collections import namedtuple

import torch

def load_ext(name, funcs):
    ext = importlib.import_module('bench2driveMMCV.' + name)
    for fun in funcs:
        assert hasattr(ext, fun), f'{fun} miss in module {name}'
    return ext

def check_ops_exist():
    ext_loader = pkgutil.find_loader('bench2driveMMCV._ext')
    return ext_loader is not None
