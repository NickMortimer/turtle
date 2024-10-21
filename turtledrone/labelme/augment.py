import cv2
import numpy as np
from copy_paste import CopyPaste
from coco import CocoDetectionCP
from visualize import display_instances
import albumentations as A
import random
from matplotlib import pyplot as plt


transform = A.Compose([
        A.RandomScale(scale_limit=(-0.9, 1), p=1), #LargeScaleJitter from scale of 0.1 to 2
        A.PadIfNeeded(256, 256, border_mode=1), #pads with image in the center, not the top left like the paper
        A.RandomCrop(256, 256),
        CopyPaste(blend=True, sigma=1, pct_objects_paste=0.8, p=1.) #pct_objects_paste is a guess
    ], bbox_params=A.BboxParams(format="coco", min_visibility=0.05)
)

data = CocoDetectionCP(
    '/media/mor582/Storage/Nick/surveys/train/yolo/train_best', 
    '/home/mor582/turtle/runs/labelme2coco/dataset.json', 
    transform
)

# typical albumentations transform
index = 1 #random.randint(0, len(data))
img_data = data[index]
image = img_data['image']
masks = img_data['masks']
bboxes = img_data['bboxes']

cpy_pst = CopyPaste(blend=True, sigma=1, pct_objects_paste=0.8, p=1., always_apply=True, max_paste_objects=5) 
out = cpy_pst(image=image, masks=[], bboxes=[], paste_image=paste_image, paste_masks=paste_masks, paste_bboxes=paste_bboxes)
