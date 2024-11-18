import os
from inference import run_coralscop, visualize_predictions

# Set required directories
img_path = "imgs"
json_path = "imgs_json_output"
output_path = "vis_mask_output"
weights_path = "checkpoints/vit_b_coralscop.pth"

# Run Inference with segmentation model and generate images with coral masks
run_coralscop(test_img_path=img_path, output_path=json_path, checkpoint_path=weights_path)
visualize_predictions(img_path=img_path, json_path=json_path, output_path=output_path)

os.chmod(json_path, 0o777)
os.chmod(output_path, 0o777)
