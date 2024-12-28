from network import *
from dataset import *
from torch_util import *

# Basic config
input_bits, output_bits = 16, 16
seqlen = 20
# prng_name = "xoroshiro128plus"
# prng_name = "vacuum_fluctuation"
prng_name = "lehmer64"
# prng_name = "xoroshiro128plusplus"
# prng_name = "LCG28"
# prng_name = "PCG32"
lr = 3e-4
weight_decay = 0
batch_size = 1024
seed = 10
double_precision = False
# model_name = "lehmerLearnedCracker"
# model_name = "ViT+GPT2"
model_name = "lehmerFC"
predict_list = []
net_prototype = ResFC
dataset_prototype = Dataset
# dataset_prototype = IndexDataset
train_split = (0, 0.7)
test_split = (0.7, 1)
# mt_data_dir = f"./data/rawdata-5-16-combine1G_150m.dat"
mt_data_dir = f"./data/lehmer64_24.dat"
register_data_dir = None
data_size = 24
num_workers = 32
# Auxiliary config
num_epochs = 20000
test_epoch = 1
num_gpus = 8
print_freq = int((2 ** data_size) * 0.7 / batch_size / num_gpus / 5)
clip_weight = False
write = True
loss_func = mtrng_loss
# loss_func = transformer_loss
regularization_weight = 0
# Save logs & model
use_multiepochs_dataloader = False
comment = f"{model_name}_seqlen={seqlen}_{prng_name}_ddp"
save_path = f"./model/{model_name}_{prng_name}_{batch_size}_{num_epochs}_{lr}_{comment}.ckpt"
log_dir = f"./logs/{model_name}_{prng_name}_{batch_size}_{num_epochs}_{lr}_{comment}/"
