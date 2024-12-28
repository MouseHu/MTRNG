import pickle as pkl
import math
import numpy as np

log_data_size = 20
data_size = 2 ** log_data_size


def lehmer64(datasize=data_size):
    g_lehmer64_state = 574389759345345 & 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF  # 128-bit state

    def _random():
        nonlocal g_lehmer64_state
        g_lehmer64_state *= 0xda942042e4dd58b5
        g_lehmer64_state &= 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
        # print(g_lehmer64_state, g_lehmer64_state >> 64)
        return g_lehmer64_state, g_lehmer64_state >> 64

    full_data = [_random() for _ in range(data_size)]
    state = [x[0] for x in full_data]
    data = [x[1] for x in full_data]
    return data, state


data, state = lehmer64()
# with open('../data/lehmer64_{}.pkl'.format(log_data_size), 'wb') as f:
#     pkl.dump(data, f)
print(math.log2(max(data)))
print(math.log2(max(state)))
print(data[:3])
print(state[0])
data = np.array(data, dtype=np.uint64)
print(data[0])
data.tofile('../data/lehmer64_{}.dat'.format(log_data_size))
with open('../data/lehmer64_state_{}.pkl'.format(log_data_size), 'wb') as f:
    pkl.dump(state, f)
