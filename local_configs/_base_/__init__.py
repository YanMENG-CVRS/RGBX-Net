import os
import os.path as osp
import time
from easydict import EasyDict as edict

C = edict()
config = C

C.root_dir = "datasets"
C.abs_dir = osp.realpath(".")
