import glob
import json
import os
import time
import argparse

import cv2
import matplotlib.pyplot as plt
from inference import get_coralscop, get_inference_metrics, run_coralscop

parser = argparse.ArgumentParser()
parser.add_argument(
    "--test_img_path", type=str, default="imgs", help="The test image path"
)
parser.add_argument(
    "--json_mask_output",
    type=str,
    default="json_output",
    help="Path to save masks in JSON format",
)
parser.add_argument(
    "--output_path",
    type=str,
    default="vis_mask_output",
    help="Path to save mask overlayed images",
)
parser.add_argument(
    "--checkpoint_path",
    type=str,
    default="checkpoints/vit_b_coralscop.pth",
    help="Path to the checkpoint",
)
parser.add_argument(
    "--gpu_index", type=int, default=0, help="Index of the GPU device to be used"
)

args = parser.parse_args()

# Set required directories
img_path = args.test_img_path
json_output_path = args.json_mask_output
output_path = args.output_path
checkpoint_path = args.checkpoint_path
gpu_index = args.gpu_index

images = []
# Load the CoralSCOP model
model = get_coralscop(checkpoint_path=checkpoint_path, gpu=gpu_index)

# Read Images
for files in sorted(glob.glob(os.path.join(img_path, "*.*"))):
    image = cv2.imread(files)
    images.append(image)

# Segment images with CoralSCOP
start_time = time.time()
masks, masked_images = run_coralscop(images, model)
end_time = time.time()
print("Total time taken for predictions:", end_time - start_time)

# Get Metrics on generated images
iou, sta = get_inference_metrics(masks)
print("Average IoU for all images is: ", sum(iou.values()) / len(iou.values()))
print(
    "Average Stability score for all images is: ", sum(sta.values()) / len(sta.values())
)


# Save the masks in JSON
if not os.path.exists(json_output_path):
    os.mkdir(json_output_path)
for i in range(len(masks)):
    json_file_name = os.path.join(json_output_path, f"image_{i}" + ".json")
    with open(json_file_name, "w") as fp:
        json.dump(masks[i], fp)

# Save the images
for i in range(len(masked_images)):
    plt.figure(figsize=(20, 20))
    plt.imshow(masked_images[i])
    plt.axis("off")
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    plt.savefig(os.path.join(output_path, f"image_{i}"), bbox_inches="tight")
    plt.gcf().clear()
    plt.close()
