from .. import *

# Dataset config
"""Dataset Path"""
C.dataset_name = "S1S2Water"
C.root = '/home/datasets/hyperspectral_dataset/S1S2Water'
C.num_train_imgs = 12230
C.num_eval_imgs = 5497
C.num_classes = 2

C.background = 255
C.image_height = 512
C.image_width = 512
C.class_names = ['Background', 'Water']
