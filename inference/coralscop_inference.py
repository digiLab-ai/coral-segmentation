import os.path
import pycocotools.mask as mask_util
import numpy as np
import cv2
import json
import glob
import os
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

def run_coralscop(test_img_path,
                  output_path,
                  checkpoint_path,
                  model_type='vit_b',  
                  iou_threshold=0.72, 
                  sta_threshold=0.62,
                  point_number=48,
                  batch_size=1,
                  gpu=0,
                  image_size=1024
                  ):
    device = "cuda"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    sam_checkpoint = checkpoint_path
    model_type = model_type
    if not os.path.exists(output_path):
        os.mkdir(output_path)
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
    for files in sorted(glob.glob(os.path.join(test_img_path,"*.*"))):
        image = cv2.imread(files)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[0], image.shape[1]
        p,n=os.path.split(files)
        if os.path.exists(os.path.join(output_path, n[:-4] + ".json")):
            continue
        try:
            masks = mask_generator.generate(image)
        except:
            continue
        print(n)
        print(len(masks))
        output_json = {}
        img_json = {}
        img_json['image_id'] = 0
        img_json['width'] = width
        img_json['height'] = height
        img_json['file_name'] = n
        output_json['image'] = img_json
        out_anno = []
        anno_id = 0
        for tmp in masks:
            anno_json = {}
            seg = tmp['segmentation']
            fortran_ground_truth_binary_mask = np.asfortranarray(seg)
            compressed_rle = mask_util.encode(fortran_ground_truth_binary_mask)
            compressed_rle['counts'] = str(compressed_rle['counts'], encoding="utf-8")
            anno_json['segmentation'] = compressed_rle
            anno_json['bbox'] = tmp['bbox']
            anno_json['area'] = tmp['area']
            anno_json['bbox'] = tmp['bbox']
            anno_json['predicted_iou'] = tmp['predicted_iou']
            anno_json['crop_box'] = tmp['crop_box']
            anno_json['stability_score'] = tmp['stability_score']
            anno_json['point_coords'] = tmp['point_coords']
            anno_json['id'] = anno_id
            out_anno.append(anno_json)
            anno_id += 1
        output_json['annotations'] = out_anno
        with open(os.path.join(output_path, n.replace(".jpg",".json")), "w") as fp:
            json.dump(output_json, fp)
