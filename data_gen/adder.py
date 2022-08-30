import pickle as pkl
import math
import numpy as np
import randomgen

log_data_size = 24
data_size = 2 ** log_data_size

RNG = randomgen.Xoroshiro128(seed=0)
rand_A = RNG.random_raw(data_size, output=True)
rand_B = RNG.random_raw(data_size, output=True)
print(type(rand_B))
data = np.array([rand_A, rand_B, rand_B + rand_A]).transpose()
data = data.astype(np.uint64)
# data = [[rand_A[i], rand_B[i], rand_A[i] + rand_B[i]] for i in range(data_size)]
print(data[0])
# data = np.array(data, dtype=np.uint64).reshape(-1)
# data.tofile('../data/adder_{}.dat'.format(log_data_size))
