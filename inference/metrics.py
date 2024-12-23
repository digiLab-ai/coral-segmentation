import os
import glob
import json

def get_inference_metrics(img_path,
                          json_path,
                          output_path):
    json_path = json_path
    img_path = img_path
    output_path = output_path
    pred_metrics_iou = {}
    pred_metrics_sta = {}
    for files in glob.glob(os.path.join(json_path,"*.json")):
        with open(files, "r", encoding='utf-8') as f:
            aa = json.loads(f.read())
            images = aa['image']
            annotations = aa['annotations']
            img_name=images['file_name']
            # file_name = f"metrics_{img_name}"
            iou_per_image = {}
            mask_sta_per_image = {}

            for i, ann in enumerate(annotations):
                iou_per_image[f"iou_{i}"] = ann["predicted_iou"]
                mask_sta_per_image[f"sta_value_{i}"] = ann["stability_score"]

            avg_pred_iou = sum([value for _, value in iou_per_image.items()]) / len(iou_per_image.keys())
            avg_mask_sta_score = sum([value for _, value in mask_sta_per_image.items()]) / len(mask_sta_per_image.keys())
            # print(f"Average IoU for image {img_name} is:", avg_pred_iou)
            # print(f"Average mask stability score for image {img_name} is:", avg_mask_sta_score)
            pred_metrics_iou[f"iou_{img_name}"] = avg_pred_iou
            pred_metrics_sta[f"mask_sta_score_{img_name}"] = avg_mask_sta_score

    print("Average IoU for all images is: ", sum(pred_metrics_iou.values())/len(pred_metrics_iou.values()))
    print("Average Stability score for all images is: ", sum(pred_metrics_sta.values())/len(pred_metrics_sta.values()))
    return pred_metrics_iou, pred_metrics_sta

