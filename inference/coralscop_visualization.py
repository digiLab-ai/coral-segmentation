import os
import glob
import itertools
import json
from typing import Dict, List

import cv2
from detectron2.data import MetadataCatalog
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import pycocotools.mask as mask_util

from inference.visualizer import Visualizer

metadata = MetadataCatalog.get('coco_2017_train_panoptic')
color_mappings = {0: [224, 0, 0], 1: [138, 43, 226]}

for k in color_mappings:
    for i in range(3):
        color_mappings[k][i] /= 255.0

def bunch_coords(coords):
    coords_trans = []
    for i in range(0, len(coords) // 2):
        coords_trans.append([coords[2 * i], coords[2 * i + 1]])
    return coords_trans

def unbunch_coords(coords):
    return list(itertools.chain(*coords))

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    for ann in sorted_anns[:128]:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = color_mappings[1]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def add_boundary(node_coods,ax):
    polygon = Polygon(node_coods, closed=False, edgecolor='r')
    ax.add_patch(polygon)

def visualize_predictions(
    image_list: List[np.ndarray],
    masks_list: List[Dict],
    alpha: float = 0.4,
    min_area: int = 4096) -> List[np.ndarray]:
    """
    Function to visualize segmentation masks on the original images

    Parameters
    __________
    image_list: List[np.ndarray]
        List of images in the numpy array format
    masks_list: List[Dict]
        List of masks where masks for every image is a dictionary
    alpha: float
        Transparency of the mask
    min_area: int 
        Minimum area of the mask

    Returns
    _______
    List[np.ndarray]
        List of masked images
    """
    alpha = alpha
    label_mode = '1'
    anno_mode = ['Mask']
    min_area=min_area
    masked_images = []
    for i in range(len(image_list)):
            aa = masks_list[i]
            annotations = aa['annotations']
            image = cv2.cvtColor(image_list[i], cv2.COLOR_BGR2RGB)
            visual = Visualizer(image, metadata=metadata)
            label = 1
            mask_map = np.zeros(image.shape, dtype=np.uint8)
            for i, ann in enumerate(annotations):
                if ann['segmentation']==[]:
                    continue
                mask = mask_util.decode(ann['segmentation'])
                if np.sum(mask)<min_area:
                    continue
                color_mask=color_mappings[1]
                demo = visual.draw_binary_mask_with_number(mask,color=color_mask,edge_color=[1.0,0,0], text="", label_mode=label_mode, alpha=alpha,
                                                           anno_mode=anno_mode)
                mask_map[mask == 1] = label
            im = demo.get_image()
            masked_images.append(im)

    return masked_images
