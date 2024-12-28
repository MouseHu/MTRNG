import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# from configs.mt import *
from configs.config_adder import *
from dataset.dataset import MTDataset, TemperDataset, HalfCrackerDataset, LehmerForwardDataset, LehmerBackwardDataset, \
    LehmerDataset
from dataset.tmp_dataset import AdderDataset
from network.fc import Temper, Cracker, CNNTwister, ResCNNTemper, LehmerForward, Adder, ResFC
from network.nalu import NeuralAdder, NeuralMultiplier, LehmerCracker
from network.nalu_vec import LehmerLearnedCracker, MultiplierCracker, LehmerSimpleCracker
from network.rnn import Multiplier
from network.attention import AttentionTwister
from torch_util import *
from dataset.dataloader import MultiEpochsDataLoader

print("train type ", train_type)
writer = None
# Define network, dataset and optimizer
train_dataset = LehmerDataset(mt_data_dir, register_data_dir, split=(0, 0.7))
test_dataset = LehmerDataset(mt_data_dir, register_data_dir, split=(0.7, 1))
model = LehmerSimpleCracker().cuda()

optimizer = optim.Adam(model.parameters(), lr=lr)
# optimizer = optim2.RAdam(model.parameters(), lr=lr)


if write:
    writer = SummaryWriter(log_dir)


# Testing Loop
def test(epoch, test_loader, total_steps):
    # model.eval()
    test_loss = 0.0
    test_correct = 0.0
    num_batches = 0
    cum_info = {}
    for i, data in tqdm(enumerate(test_loader)):
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        inputs = 1. * inputs

        outputs = model(inputs)
        loss, correct, info = mtrng_loss(model, outputs, labels, l1_coef=weight_decay, predict_list=predict_list)

        # print(info["correct_per_sample"])
        print(info["correct_per_cat"])
        num_batches += 1


if __name__ == '__main__':
    train_loader = MultiEpochsDataLoader(train_dataset, batch_size=batch_size, pin_memory=False,
                                         shuffle=False, num_workers=num_workers)
    test(1, train_loader, 0)
