import numpy as np

m_ = 28
nbits = 8
log_data_size = 24
data_size = 2 ** log_data_size


class LCG(object):
    def __init__(self, m, a, c, current=10, nbits=8):
        self.a = a
        self.m = 2**m
        self.c = c
        self.nbits = nbits
        self.current = current

    def __next__(self):
        self.current = (self.a * self.current + self.c) % self.m
        return np.uint8((float(self.current) / self.m) * (2 ** self.nbits))


lcg = LCG(m_, 1103515245, 12345, nbits=nbits)
data = np.array([next(lcg) for _ in range(data_size)], dtype=np.uint8)
print(data)
data.tofile(f'../data/LCG{m_}_{log_data_size}.dat')
