import numpy as np
from dataset.dataset import binary, LehmerForwardDataset, LehmerBackwardDataset

# x = np.arange(10, dtype=np.uint8)
# print(x)
# print(binary(x).reshape(-1, 8))
# print(binary(x).shape)
# x = 10
# print(binary(x))
# x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# print(binary(x))

def to_int(m):
    return int(''.join([str(a) for a in m.numpy()]), 2)
# forward_dataset = LehmerForwardDataset(data_dir='../data/lehmer64_24.dat',
#                                        state_data_dir='../data/lehmer64_state_24.pkl', split=[0.0, 1.0])

backward_dataset = LehmerBackwardDataset(data_dir='../data/lehmer64_24.dat',
                                         state_data_dir='../data/lehmer64_state_24.pkl', split=[0.0, 1.0])
x, y = backward_dataset[0]
print(len(y))
print(to_int(x[0]))
print(to_int(x[1]))
print(to_int(x[2]))
print(to_int(y))

# x, y = backward_dataset[0]
# print(x, y)
