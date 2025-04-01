# standard imports
import os

# package imports
import torch

# global inference device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# path to model checkpoints
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "checkpoints")