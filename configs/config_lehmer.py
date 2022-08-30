import multiprocessing as mp

input_bits, output_bits = 128, 128
lr = 1e-3
weight_decay = 0
batch_size = 8192
num_epochs = 20000
test_epoch = 5
data_size = 24
print_freq = int((2 ** data_size) * 0.7 / batch_size / 5)

mt_data_dir = f"./data/lehmer64_{data_size}.dat"
register_data_dir = f"./data/lehmer64_state_{data_size}.pkl"
train_type = 5
use_autoencoder = False
num_workers = min(64, mp.cpu_count())
# train type = 0 train temper; = 1 train inverse temper; =2 train register; =3 train whole
predict_list = None
train_type_name = ["inverse_temper", "temper", "twister", "cracker", "temper_lehmer_forward", "lehmer_total"][
    train_type]
comment = f"{train_type_name}_lehmer_multiplier"
write = True

save_path = f"./model/{train_type}_{batch_size}_{num_epochs}_{lr}_{data_size}_{comment}.ckpt"
log_dir = f"./logs/{train_type}_{batch_size}_{num_epochs}_{lr}_{data_size}_{comment}/"
