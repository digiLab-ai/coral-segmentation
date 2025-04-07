ARG UBUNTU_VERSION=22.04
ARG NVIDIA_CUDA_VERSION=12.1.0

# Retrieve image with required CUDA and OS version
FROM nvidia/cuda:${NVIDIA_CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}

# Install system-level dependencies
RUN apt-get update && \
    apt-get install -y git \
    gcc-11 \
    g++-11 \
    libgl1\
    libgl1-mesa-glx \ 
    libglib2.0-0 \
    python3.10 \
    pip

# Clone repository and set working directory
RUN git clone https://github.com/digiLab-ai/coral-segmentation.git
# COPY inference/ segment_anything/ main.py 

# Install requirements to run model
RUN pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
RUN pip install opencv-python

# Install detectron2 for mask utilities
RUN git clone https://github.com/facebookresearch/detectron2.git
RUN pip install -e detectron2

# Run test script for inference
ENTRYPOINT ["python3", "main.py"]


