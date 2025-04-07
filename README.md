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

This is a repository that is meant to run segmentation on videos. However, the underlying model can only process images and not videos. 
So, before trying to run inference with the model we can pre-process the video to obtain a set of images from the video. To do this, run the following command:

```
python preprocess.py --video_path=<PATH_TO_VIDEO> --img_path=<OUTPUT_PATH_TO_WRITE_IMAGES>
```

The outputs from the pre-processing step are a set of images stored in the specified directory. To segment the pre-processed images, run the following command:

```
python main.py --test_img_path=<PATH_TO_IMAGES> --json_mask_output=<PATH_TO_STORE_MASKS_IN_JSON> --output_path=<PATH_TO_STORE_MASKED_IMAGES> --checkpoint_path=<PATH_TO_MODEL_CHECKPOINT>
```

## Acknowledgement

+ [CoralSCOP](https://github.com/zhengziqiang/CoralSCOP.git) Thanks for their contributions to the whole Marine Science and Computer Vision community!

## Acknowledgement

We sincerely thank Ziqiang Zheng for his inputs on the inference codes for CoralSCOP.
