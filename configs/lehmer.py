input_bits, output_bits = 64, 128
lr = 1e-3
weight_decay = 0
batch_size = 16384
num_epochs = 20000
print_freq = 180
test_epoch = 5
data_size = 24

mt_data_dir = f"./data/lehmer64_{data_size}.dat"
# mt_data_dir = "/data/qrng/mtrng_24.dat"
register_data_dir = f"./data/lehmer64_state_{data_size}.pkl"
train_type = 4
use_autoencoder = False
num_gpus = 8
# train type = 0 train temper; = 1 train inverse temper; =2 train register; =3 train whole
predict_list = [9, 27, 31]
train_type_name = ["inverse_temper", "temper", "twister", "cracker", "temper_lehmer_forward"][train_type]
comment = f"{train_type_name}_lehmer"
write = True


save_path = f"./model/{train_type}_{batch_size}_{num_epochs}_{lr}_{data_size}_{comment}.ckpt"
log_dir = f"./logs/{train_type}_{batch_size}_{num_epochs}_{lr}_{data_size}_{comment}/"

