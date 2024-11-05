# Install CUDA version 12.1 and Torch with corresponding wheel
conda install -c nvidia/label/cuda-12.1.0 cuda-toolkit  
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python

# Install GCC for the conda environment (any version less than 12) and install detectron2
conda install -c conda-forge gxx=11.4.0
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2