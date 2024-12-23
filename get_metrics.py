import argparse
from inference import get_inference_metrics

parser = argparse.ArgumentParser()
parser.add_argument("--test_img_path", type=str, default="imgs", help="the test image path")
parser.add_argument("--json_output_path", type=str, default="imgs_json_output", help="path to save json file")
parser.add_argument("--output_path", type=str, default="vis_mask_output", help="path to save mask overlayed images")

args = parser.parse_args()
img_path = args.test_img_path
json_path = args.json_output_path
output_path = args.output_path

iou_metric, sta_metric = get_inference_metrics(img_path=img_path, json_path=json_path, output_path=output_path)
