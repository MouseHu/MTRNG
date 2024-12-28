import numpy as np
from constants import mt_data_dir
from dataset.dataset import binary
import os

mt_data = np.fromfile(os.path.join(os.curdir, "../data/mtrng_20.dat"), dtype=np.uint32)
binary_mt_data = binary(mt_data)
correct_rate = []
for i in range(454):
    guess = np.random.randint(0, 2, binary_mt_data.shape)
    correct = np.sum(binary_mt_data == guess)
    total = binary_mt_data.shape[0]
    correct_rate.append(correct / total)
    # print(correct, total, correct / total)

print(np.max(correct_rate), np.var(correct_rate))
