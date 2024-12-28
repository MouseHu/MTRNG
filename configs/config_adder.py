import multiprocessing as mp
import time
from network import *
from dataset import *

input_bits, output_bits = 64, 64
lr = 1e-3
weight_decay = 0
batch_size = 1024
num_epochs = 20000
test_epoch = 5
data_size = 20
double_precision = False
print_freq = int((2 ** data_size) * 0.7 / batch_size / 5)

# mt_data_dir = f"./data/adder_{data_size}.dat"
mt_data_dir = f"./data/lehmer64_{data_size}.dat"
register_data_dir = f"./data/lehmer64_state_{data_size}.pkl"

# train_type = 4
model_name = "lehmerCNNCrackerNoActivation"
net_prototype = LehmerCNNCracker
dataset_prototype = LehmerDataset
num_workers = min(64, mp.cpu_count())
# train type = 0 train temper; = 1 train inverse temper; =2 train register; =3 train whole
# predict_list = [_ for _ in range(52, 64)]
predict_list = None
# predict_list = [_ for _ in range(52, 64)]
train_type_name = "neural_multiplier_promising"
comment = f"{train_type_name}"
write = False

save_path = f"./model/{model_name}_{batch_size}_{num_epochs}_{lr}_{data_size}_{comment}.ckpt"
log_dir = f"./logs/{model_name}_{batch_size}_{num_epochs}_{lr}_{data_size}_{comment}/{int(time.time())}"
