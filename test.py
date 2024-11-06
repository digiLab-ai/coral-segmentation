from inference import run_coralscop, visualize_predictions

run_coralscop(test_img_path="demo_imgs", output_path="demo_imgs_json_output", checkpoint_path="checkpoints/vit_b_coralscop.pth")

visualize_predictions(img_path="demo_imgs", json_path="demo_imgs_json_output", output_path="demo_vis_mask_output")