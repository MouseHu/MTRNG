from network import *
from dataset import *
# Basic config
input_bits, output_bits = 32, 32
seqlen = 50
lr = 1e-3
weight_decay = 0
batch_size = 1024
seed = 10
data_size = 25
net_prototype = MTCracker
dataset_prototype = MTDataset
mt_data_dir = f"./data/mtrng_{data_size}.dat"
predict_list = [9, 27, 31]
double_precision = False
num_workers = 32

custom_optimizer = False
trainer_config = None
train_split = 0.97
total_split = 1
# Auxiliary config
num_epochs = 20000
print_freq = 924
test_epoch = 5
num_gpus = 8

# Save logs & model

comment = f"cracker_seqlen={seqlen}_ddp_test"
save_path = f"./model/cracker_{batch_size}_{num_epochs}_{lr}_{data_size}_{comment}.ckpt"
log_dir = f"./logs/cracker_{batch_size}_{num_epochs}_{lr}_{data_size}_{comment}/"
