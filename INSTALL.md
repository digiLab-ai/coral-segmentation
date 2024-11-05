**1. Installing dependencies**

Git clone the repository and install CUDA toolkit version 12.1 and 
corresponding torch versions from the respective wheels.

Then install detectron2 for better mask visualization.

```bash
git clone https://github.com/digiLab-ai/coral-segmentation.git
conda create -n coralscop python=3.10
conda install -c nvidia/label/cuda-12.1.0 cuda-toolkit  
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python

conda install -c conda-forge gxx=11.4.0
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
```
