import os
import argparse
from inference import run_coralscop, visualize_predictions

parser = argparse.ArgumentParser()
parser.add_argument("--test_img_path", type=str, default="imgs", help="the test image path")
parser.add_argument("--json_output_path", type=str, default="imgs_json_output", help="path to save json file")
parser.add_argument("--output_path", type=str, default="vis_mask_output", help="path to save mask overlayed images")
parser.add_argument("--checkpoint_path", type=str, default="checkpoints/vit_b_coralscop.pth", help="path to the checkpoint")
parser.add_argument("--gpu_index", type=int, default=0, help="Index of the GPU device to be used")

args = parser.parse_args()

# Set required directories
img_path = args.test_img_path
json_path = args.json_output_path
output_path = args.output_path
weights_path = args.checkpoint_path
gpu_index = args.gpu_index

# Run Inference with segmentation model and generate images with coral masks
run_coralscop(test_img_path=img_path, output_path=json_path, checkpoint_path=weights_path, gpu=gpu_index)
visualize_predictions(img_path=img_path, json_path=json_path, output_path=output_path)

os.chmod(json_path, 0o777)
os.chmod(output_path, 0o777)
