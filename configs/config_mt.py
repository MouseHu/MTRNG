input_bits, output_bits = 32, 32
seqlen = 624
lr = 1e-3
weight_decay = 0
batch_size = 1024
num_epochs = 20000
print_freq = 180
test_epoch = 5
data_size = 24
total_steps = 5730 * 20
mt_data_dir = f"./data/mtrng_{data_size}.dat"
# mt_data_dir = "/data/qrng/mtrng_24.dat"
register_data_dir = f"./data/register_mtrng_{data_size}.dat"
train_type = 3
use_autoencoder = False
num_workers = 128
# train type = 0 train temper; = 1 train inverse temper; =2 train register; =3 train whole
predict_list = None
train_type_name = ["inverse_temper", "temper", "twister", "cracker", "halfcracker"][train_type]
comment = f"{train_type_name}_adam_seqlen={seqlen}_ddp_show_all_rank_2"
write = False


save_path = f"./model/{train_type}_{batch_size}_{num_epochs}_{lr}_{data_size}_{comment}.ckpt"
log_dir = f"./logs/{train_type}_{batch_size}_{num_epochs}_{lr}_{data_size}_{comment}/"

inverse_temper_dir = "./model/0_1024_20000_0.0003_20_temper_save.ckpt"
temper_dir = "./model/1_128_20000_0.0003_20_temper_resnet_large.ckpt"
twister_dir = "./model/2_128_20000_0.0003_20_twister_large_resnet.ckpt"

cracker_dir = "./model/3_1024_20000_0.01_24_cracker_adam_seqlen=624_ddp_what.ckpt"

