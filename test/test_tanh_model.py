import torch
from dataset.dataset import Dataset, TemperDataset
from constants import *
from network.fc import *
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from utils import l1_norm

model_dir = "./model/3_128_20000_0.0003_20_cracker_non_loading_tanh.ckpt"
dataset = TemperDataset(register_data_dir, mt_data_dir, split=(0, 1))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=False, num_workers=16)

model = Cracker().cuda()
model.load_fully(model_dir)
# model.load(temper_dir, inverse_temper_dir, twister_dir)
print(l1_norm(model.temper), l1_norm(model.inverse_temper), l1_norm(model.twister))
model.eval()
register_data = np.zeros((len(dataset), input_bits))
inverse_data = np.zeros((len(dataset), input_bits))
for i, data in tqdm(enumerate(dataloader)):
    # get the inputs; data is a list of [inputs, labels]
    # inputs, labels = data
    inputs, labels = data
    inputs = 2 * inputs - 1.
    labels = 2 * labels - 1.
    start = i * batch_size
    end = min((i + 1) * batch_size, len(dataset))
    inputs, labels = inputs.cuda(), labels.cuda()
    register_batch = torch.tanh(model.inverse_temper(inputs))
    inverse_batch = torch.tanh(model.temper(labels))
    register_data[start:end] = register_batch.detach().cpu().numpy()
    inverse_data[start:end] = inverse_batch.detach().cpu().numpy()

# all_data = [[(j, register_data[i, j]) for j in range(input_bits)] for i in range(len(dataset))]
# all_data = sum(all_data, [])
plt.figure(figsize=(20, 16))
for i in range(input_bits):
    plt.scatter(i * np.ones(len(register_data[::300, i])), register_data[::300, i])
# plt.show()
plt.savefig("test_tanh_model.png")
plt.close()
print("Fig 1")

plt.figure(figsize=(20, 16))
for i in range(input_bits):
    plt.scatter(i * np.ones(len(inverse_data[::300, i])), inverse_data[::300, i])
# plt.show()
plt.savefig("test_tanh_model_inverse.png")
plt.close()
print("Fig 2")
