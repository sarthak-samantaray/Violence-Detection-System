import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_SHAPE = 165
HIDDEN_SIZE = 128
NUM_CLASSES = 2
