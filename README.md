## Coral Segmentation

This repository contains the code for the segmentation model used in the Automated Coral
Measurement Pipeline created as part of KAUST WP4

## Getting Started
### Installation

**1. Create and setup environment**

Git clone the repository, create a python conda environment and activate it with the following commands:

```bash
git clone https://github.com/digiLab-ai/coral-segmentation.git
conda create -n coralscop python=3.10
conda activate coralscop
```

The shorter way to install the dependencies is to just run:

```bash
sh setup.sh
```
This will install all the required dependencies for running inference with the segmentation model.
However if there are any issues with the setup script, the straightforward way is to follow the requirements installation guide below.

**2. Installing dependencies (manually)**

Start by installing CUDA Toolkits inside the environment locally and install pytorch from the corresponding CUDA wheel as follows:

```bash
conda install -c nvidia/label/cuda-12.1.0 cuda-toolkit  
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
```

Install some additional dependencies including `opencv-python` and `detectron2` for mask manipulation and visualization 
with the following commands (install GCC complier inside the conda environment to build detectron2 from source):

```bash
pip install opencv-python

conda install -c conda-forge gxx=11.4.0
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
```


**3. Download the pre-trained weights**

We provide the trained [CoralSCOP model](https://www.dropbox.com/scl/fi/pw5jiq9oc8e8kvkx1fdk0/vit_b_coralscop.pth?rlkey=qczdohnzxwgwoadpzeht0lim2&st=actcedwy&dl=0) (ViT-B backbone) for users to generate the coral reef masks based on their own coral reef images. Please download the model and put it inside a folder that is named `checkpoints`.


**4. Testing and visualization**

Testing the model based on your own coral reef images. The outputs will be saved in JSON format (COCO RLE). Please specify the output folder:

```
python test.py --model_type vit_b --checkpoint_path ./checkpoints/vit_b_coralscop.pth --iou_threshold 0.72 --sta_threshold 0.62 --test_img_path ./demo_imgs/ --output_path ./demo_imgs_output --gpu 0 --point_number 32
```

`model_type` indicates the backbone type; `checkpoint_path`: model checkpoint path, please change to your path; `iou_threshold`: predict iou threshhold, masks with predicted iou lower than 0.72 will be removed;`sta_threshold`: stability score threshhold, masks with stability score lower than 0.62 will be removed; `test_img_path`: your testing image path; `output_dir`: output path for saving the generated jsons; `gpu`: which gpu to use.
   
Visualize the generated jsons :

```
python coralscop_visualization.py --img_path ./demo_imgs/ --json_path ./demo_imgs_output/ --output_path ./vis_demo 
```
`img_path`: same as the `test_img_path`, the testing images; `json_path`: same sa the `output_dir`, the path for saving generated json; `output_path`: the path for saving the images with visualizations. 

## Acknowledgement

+ [CoralSCOP](https://github.com/zhengziqiang/CoralSCOP.git) Thanks for their contributions to the whole Marine Science and Computer Vision community!

## Acknowledgement

We sincerely thank Ziqiang Zheng for his inputs on the inference codes for CoralSCOP.
