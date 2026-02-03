# Copyright (c) Facebook, Inc. and its affiliates.

# CRITICAL: Import cpu_compat FIRST to monkey patch torch.cuda.amp
# This must happen before any timm imports
from . import cpu_compat  # noqa: F401

from . import data  # register all new datasets
from . import modeling

# config
from .config import *

# models
from .oneformer_model import OneFormer