import numpy as np
import matplotlib
from scipy.stats import entropy

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from dataset.dataset import read_data


data_dir = "/home/hh/mtrng/data/rawdata-5-16-combine1G_150m.dat"
split = (0.07,0.1)

data = read_data(data_dir,12)
scaled_split = [int(len(data) * p) for p in split]
data = data[scaled_split[0]:scaled_split[1]]
bincount = np.bincount(data)

print(bincount)
print(entropy(bincount))
print(np.max(bincount),data.shape)
print(np.max(bincount)/np.sum(bincount))
print(len(bincount), data.shape, np.sum(bincount))


plt.plot(bincount)
plt.show()


