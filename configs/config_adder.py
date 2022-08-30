import multiprocessing as mp
import time

input_bits, output_bits = 64, 64
lr = 1e-3
weight_decay = 0
batch_size = 4096
num_epochs = 20000
test_epoch = 5
data_size = 20
print_freq = int((2 ** data_size) * 0.7 / batch_size / 5)

# mt_data_dir = f"./data/adder_{data_size}.dat"
mt_data_dir = f"./data/lehmer64_{data_size}.dat"
register_data_dir = f"./data/lehmer64_state_{data_size}.pkl"

train_type = 4
num_workers = min(64, mp.cpu_count())
# train type = 0 train temper; = 1 train inverse temper; =2 train register; =3 train whole
predict_list = None
train_type_name = "neural_multiplier"
comment = f"{train_type_name}"
write = True

save_path = f"./model/{train_type}_{batch_size}_{num_epochs}_{lr}_{data_size}_{comment}.ckpt"
log_dir = f"./logs/{train_type}_{batch_size}_{num_epochs}_{lr}_{data_size}_{comment}/{int(time.time())}"
