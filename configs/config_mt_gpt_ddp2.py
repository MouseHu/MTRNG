from network import *
from dataset import *

from network.utils import CfgNode as CN

# Basic config
input_bits, output_bits = 32, 32
seqlen = 3
lr = 1e-3
weight_decay = 0
batch_size = 256
seed = 10
data_size = 25
train_split = 0.097
total_split = 0.1

gpt_config = GPT.get_default_config()
gpt_config.model_type = 'gpt-mini'
gpt_config.memory_efficient = True
gpt_config.causal = True
gpt_config.vocab_size = 2 ** 16
gpt_config.block_size = seqlen

custom_optimizer = True
trainer_config = CN()
trainer_config.learning_rate = lr
trainer_config.betas = (0.9, 0.95)
trainer_config.weight_decay = 0.1  # only applied on matmul weights
trainer_config.grad_norm_clip = 1.0

net_prototype = gpt_wrapper(gpt_config)
dataset_prototype = MTPreloadDataset
mt_data_dir = f"./data/mtrng_{data_size}.dat"
# predict_list = [9, 27, 31]
predict_list = None
double_precision = False
num_workers = 32

# Auxiliary config
num_epochs = 20000
print_freq = 100
test_epoch = 5
num_gpus = 8

# Save logs & model

comment = f"cracker_gpt_seqlen={seqlen}_ddp2_test"
save_path = f"./model/cracker_{batch_size}_{num_epochs}_{lr}_{data_size}_{comment}.ckpt"
log_dir = f"./logs/cracker_{batch_size}_{num_epochs}_{lr}_{data_size}_{comment}/"
