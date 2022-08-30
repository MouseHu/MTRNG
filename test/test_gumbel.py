import torch
from utils import *

x = torch.randn(10)

print(x)
hard_gumbel = binary_gumbel_softmax(x)

print(hard_gumbel)
