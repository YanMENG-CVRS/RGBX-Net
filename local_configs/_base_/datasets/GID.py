from .. import *

# Dataset config
"""Dataset Path"""
C.dataset_name = "GID"
C.root = '/home/datasets/hyperspectral_dataset/GID'
C.num_train_imgs = 11018
C.num_eval_imgs = 1415
C.num_classes = 6
C.binary_cls = False

C.background = 255
C.image_height = 512
C.image_width = 512
C.class_names = ['Background', 'Building', 'Farmland', 'Forest', 'Meadow', 'Water']
