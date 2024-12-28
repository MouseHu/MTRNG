import time
import torch
for i in range(1000):
    print(i)
    tensor = torch.randn((10,1))
    tensor.to('cuda')
    print(tensor)
    time.sleep(1)
