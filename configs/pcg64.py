from network import *

# Basic config
input_bits, output_bits = 64, 64
seqlen = 30
prng_name = "PCG64"
lr = 1e-3
weight_decay = 0
batch_size = 1024
seed = 10
data_size = 25
model_name = "cracker"

net_prototype = Cracker
mt_data_dir = f"./data/{prng_name}_{data_size}.dat"

# Auxiliary config
num_epochs = 20000
print_freq = 924
test_epoch = 5
num_gpus = 8

# Save logs & model

comment = f"{model_name}_seqlen={seqlen}_{prng_name}_ddp"
save_path = f"./model/{model_name}_{prng_name}_{batch_size}_{num_epochs}_{lr}_{data_size}_{comment}.ckpt"
log_dir = f"./logs/{model_name}_{prng_name}_{batch_size}_{num_epochs}_{lr}_{data_size}_{comment}/"
