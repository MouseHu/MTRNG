from network import *
from dataset import *

# Basic config
input_bits, output_bits = 128, 64
seqlen = 1
# prng_name = "xoroshiro128plus"
prng_name = "lehmer64"
# prng_name = "xoroshiro128plusplus"
# prng_name = "LCG28"
# prng_name = "PCG32"
lr = 1e-3
weight_decay = 0
batch_size = 128
seed = 10
data_size = 24
model_name = "multiplier"
predict_list = None
net_prototype = Multiplier
dataset_prototype = LehmerForwardDataset
mt_data_dir = f"./data/{prng_name}_{data_size}.dat"
register_data_dir = f"./data/{prng_name}_state_{data_size}.pkl"
num_workers = 32
# Auxiliary config
num_epochs = 20000
test_epoch = 5
num_gpus = 8
print_freq = int((2 ** data_size) * 0.7 / batch_size / num_gpus / 5)

# Save logs & model

comment = f"{model_name}_seqlen={seqlen}_{prng_name}_ddp"
save_path = f"./model/{model_name}_{prng_name}_{batch_size}_{num_epochs}_{lr}_{data_size}_{comment}.ckpt"
log_dir = f"./logs/{model_name}_{prng_name}_{batch_size}_{num_epochs}_{lr}_{data_size}_{comment}/"
