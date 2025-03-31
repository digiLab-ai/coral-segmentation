from typing import Dict, List, Tuple

def get_inference_metrics(masks_list: List[Dict]) -> Tuple[Dict, Dict]:
    """
    Function to retrieve standard metrics for segmentation mask

    Parameters
    __________
    masks_list: List[Dict]
        List of Masks information for a set of images.
        Masks generated fdor a single image is a dictionary.

    Returns
    _______
    Tuple[Dict, Dict]
        Dictionary of Average IoU threshold values for images
        Dictionary of Stability Threshold values for images
    
    """
    pred_metrics_iou = {}
    pred_metrics_sta = {}
  
    for j in range(len(masks_list)):
        aa = masks_list[j]
        annotations = aa['annotations']
        iou_per_image = {}
        mask_sta_per_image = {}

        for i, ann in enumerate(annotations):
            iou_per_image[f"iou_{i}"] = ann["predicted_iou"]
            mask_sta_per_image[f"sta_value_{i}"] = ann["stability_score"]

        # Average Metrics over all masks per image
        avg_pred_iou = sum([value for _, value in iou_per_image.items()]) / len(iou_per_image.keys())
        avg_mask_sta_score = sum([value for _, value in mask_sta_per_image.items()]) / len(mask_sta_per_image.keys())
        
        # Append to book-keeping dictionary that holds the average values across images
        pred_metrics_iou[f"iou_{j}"] = avg_pred_iou
        pred_metrics_sta[f"mask_sta_score_{j}"] = avg_mask_sta_score

    # Return the dictionaries that hold image-level metrics
    return pred_metrics_iou, pred_metrics_sta

