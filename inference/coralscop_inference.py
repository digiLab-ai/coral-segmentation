import os, os.path
from typing import List

import numpy as np
import pycocotools.mask as mask_util

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from .coralscop_visualization import visualize_predictions
from .utils import DEVICE, CHECKPOINT_PATH


def get_coralscop(
    checkpoint_path: str = CHECKPOINT_PATH,
    model_type: str = "vit_b",
    iou_threshold=0.72,
    sta_threshold=0.62,
    point_number=48,
    gpu=0,
) -> SamAutomaticMaskGenerator:
    """
    Function to load the model weights and return the mask generator object

    Parameters
    ----------
    checkpoint_path : str
        Path to the model checkpoint
    model_type : str
        Backbone architecture for the image encoder
    iou_threshold : float
        IoU threshold for mask filtering
    sta_threshold : float
        Stability score threshold for mask filtering
    point_number : int
        Number of point prompts to sample on a grid for mask generation
    gpu : int
        GPU device id

    Returns
    -------
    SamAutomaticMaskGenerator
        Mask generator object
    """
    device = DEVICE
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    sam_checkpoint = checkpoint_path
    model_type = model_type

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=point_number,
        pred_iou_thresh=iou_threshold,
        stability_score_thresh=sta_threshold,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,  # Requires open-cv to run post-processing
    )

    return mask_generator


def run_coralscop(
    images: List[np.ndarray],
    mask_generator: SamAutomaticMaskGenerator,
    batch_size=1,
    image_size=1024,
):
    """
    Function to run the mask generator on a set of images and save the output.

    Parameters
    ----------
    images: List[np.ndarray]
        List of images as numpy arrays
    mask_generator : SamAutomaticMaskGenerator
        Mask generator object
    batch_size : int
        Batch size for inference
    image_size : int
        Image size for inference
    """
    image_counter = 1
    masks_list = []
    for image in images:
        try:
            masks = mask_generator.generate(image)
        except:
            continue
        print(len(masks))
        output_json = {}
        img_json = {}
        img_json["image_id"] = 0
        img_json["height"] = image.shape[0]
        img_json["width"] = image.shape[1]
        output_json["image"] = img_json
        out_anno = []
        anno_id = 0
        for tmp in masks:
            anno_json = {}
            seg = tmp["segmentation"]
            fortran_ground_truth_binary_mask = np.asfortranarray(seg)
            compressed_rle = mask_util.encode(fortran_ground_truth_binary_mask)
            compressed_rle["counts"] = str(compressed_rle["counts"], encoding="utf-8")
            anno_json["segmentation"] = compressed_rle
            anno_json["bbox"] = tmp["bbox"]
            anno_json["area"] = tmp["area"]
            anno_json["bbox"] = tmp["bbox"]
            anno_json["predicted_iou"] = tmp["predicted_iou"]
            anno_json["crop_box"] = tmp["crop_box"]
            anno_json["stability_score"] = tmp["stability_score"]
            anno_json["point_coords"] = tmp["point_coords"]
            anno_json["id"] = anno_id
            out_anno.append(anno_json)
            anno_id += 1
        output_json["annotations"] = out_anno
        masks_list.append(output_json)
        image_counter += 1

    masked_images_list = visualize_predictions(images, masks_list)
    return masks_list, masked_images_list
