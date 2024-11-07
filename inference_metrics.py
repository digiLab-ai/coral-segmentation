from inference import get_inference_metrics

metrics = get_inference_metrics(img_path="sample_imgs", json_path="sample_imgs_json_output", output_path="sample_imgs_metrics")
print(metrics)
