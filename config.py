import torch

RESOLUTION = 1.0

PATCH_SIZE = 32
STRIDE_TRAIN = 16
STRIDE_INFER = 8

MAX_NAN_RATIO = 0.30

# Filter parameters for outlier removal
FILTER_CELL = 4.0
FILTER_THRESH = 0.6

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
