import torch
from torch import nn
import torch.nn.functional as F
from torch_util import binary_gumbel_softmax, harden, weight_init
from .cnn import CNNTemper, CNNTemper2, CNNTwister, ResCNNTemper

# ALPHA = 0.5388
ALPHA = 0.5388
NUM_FEATURES = 2048
NUM_FEATURES_TWISTER = 128
# ACTIVATION = nn.LeakyReLU(negative_slope=ALPHA)
ACTIVATION = nn.ELU()


class ResBlock(nn.Module):
    def __init__(self, shape=256, activation=F.leaky_relu):
        super(ResBlock, self).__init__()
        self.block1 = nn.Linear(shape, NUM_FEATURES)
        self.block2 = nn.Linear(NUM_FEATURES, shape)
        self.activation = activation
        self.bn1 = nn.BatchNorm1d(num_features=NUM_FEATURES, track_running_stats=False)
        self.bn2 = nn.BatchNorm1d(num_features=shape, track_running_stats=False)
        weight_init(self.block1, alpha=ALPHA)
        weight_init(self.block2, alpha=ALPHA)

    def forward(self, x):
        y = self.activation(self.bn1(self.block1(x)))
        y = self.activation(self.bn2(self.block2(y)) + x)
        return y


class Temper(nn.Module):
    def __init__(self, input_bits=32, output_bits=32):
        super().__init__()
        self.fc1 = nn.Linear(input_bits, NUM_FEATURES)
        self.fc4 = nn.Linear(NUM_FEATURES, output_bits)
        self.activation = ACTIVATION

        self.bn1 = nn.BatchNorm1d(num_features=NUM_FEATURES, track_running_stats=False)
        # init
        weight_init(self.fc1, alpha=ALPHA)
        weight_init(self.fc4, alpha=ALPHA)

    def forward(self, x):
        feature = self.activation(self.bn1(self.fc1(x)))
        output = self.fc4(feature)
        return output


class LehmerForward(nn.Module):
    MIDDLE_SHAPE = 32768
    NUM_GROUPS = 256

    def __init__(self, input_bits=32, output_bits=32):
        super().__init__()
        self.input_bits = input_bits
        self.output_bits = output_bits
        self.fc1 = nn.Linear(input_bits, self.MIDDLE_SHAPE)
        # self.fc2 = ResBlock(output_bits)
        # self.fc3 = ResBlock(output_bits)
        # self.fc4 = torch.nn.Conv1d(output_bits, output_bits,
        #                            kernel_size=(self.MIDDLE_SHAPE // output_bits,),
        #                            groups=output_bits)
        self.fc4 = nn.Linear(self.MIDDLE_SHAPE, output_bits)
        self.activation = F.elu
        self.bn1 = nn.BatchNorm1d(num_features=self.MIDDLE_SHAPE)
        # init
        weight_init(self.fc1, alpha=1)
        weight_init(self.fc4, alpha=ALPHA)

    def forward(self, x):
        batch_size = x.shape[0]
        feature = self.activation(self.bn1(self.fc1(x)))
        # feature = feature.reshape(batch_size, self.output_bits, self.MIDDLE_SHAPE // self.output_bits)
        # feature = self.fc2(feature)
        # feature = self.fc3(feature)
        output = self.fc4(feature)
        output = output.squeeze()
        return output


class LowDimTwister(nn.Module):
    def __init__(self, input_bits=32, output_bits=32, seqlen=624):
        super().__init__()
        self.seqlen = seqlen
        self.input_bits = input_bits
        self.fc1 = nn.Linear(input_bits, 4)
        self.fc3 = nn.Linear(seqlen * 4, output_bits)
        self.activation = ACTIVATION

        self.bn1 = nn.BatchNorm1d(seqlen, track_running_stats=False)
        # init
        weight_init(self.fc1, alpha=ALPHA)
        weight_init(self.fc3, alpha=ALPHA)

    def forward(self, x):
        x = x.reshape(-1, self.seqlen, self.input_bits)
        feature = self.activation(self.bn1(self.fc1(x)))
        feature = feature.reshape(-1, self.seqlen * 4)
        output = self.fc3(feature)
        return output


class Twister(nn.Module):
    def __init__(self, input_bits=32, output_bits=32, seqlen=624):
        super().__init__()
        self.fc1 = nn.Linear(input_bits * seqlen, NUM_FEATURES_TWISTER)
        self.fc2 = nn.Linear(NUM_FEATURES_TWISTER, NUM_FEATURES_TWISTER)
        self.fc3 = nn.Linear(NUM_FEATURES_TWISTER, output_bits)
        self.activation = ACTIVATION

        self.bn1 = nn.BatchNorm1d(NUM_FEATURES_TWISTER, track_running_stats=False)
        self.bn2 = nn.BatchNorm1d(NUM_FEATURES_TWISTER, track_running_stats=False)
        # init
        weight_init(self.fc1, alpha=ALPHA)
        weight_init(self.fc2, alpha=ALPHA)

    def forward(self, x):
        feature = self.activation(self.bn1(self.fc1(x)))
        # feature = self.res1(feature)
        feature = self.activation(self.bn2(self.fc2(feature)))
        output = self.fc3(feature)
        return output


class Adder(nn.Module):
    def __init__(self, seqlen):
        super().__init__()
        self.seqlen = seqlen
        self.h = 1024
        self.g = 32
        self.fc1 = nn.Linear(seqlen, self.h)
        self.fc2 = nn.Linear(self.h // self.g, self.h)
        self.fc3 = nn.Linear(self.h // self.g, self.h)
        self.fc4 = nn.Linear(self.h // self.g, 1)
        self.bn1 = nn.BatchNorm1d(self.h)
        self.bn2 = nn.BatchNorm1d(self.h)
        self.bn3 = nn.BatchNorm1d(self.h)
        self.activation = torch.logsumexp

    def forward(self, x):
        # print(x.shape)
        x = x.reshape(-1, self.seqlen)
        x = (self.bn1(self.fc1(x)))
        x = self.activation(x.reshape(-1, self.h // self.g, self.g), dim=-1)
        x = (self.bn2(self.fc2(x)))
        x = self.activation(x.reshape(-1, self.h // self.g, self.g), dim=-1)
        x = (self.bn3(self.fc3(x)))
        x = self.activation(x.reshape(-1, self.h // self.g, self.g), dim=-1)
        x = self.fc4(x)
        return x


class Cracker(nn.Module):
    def __init__(self, input_bits=32, output_bits=32, seqlen=624):
        super().__init__()
        self.seqlen = seqlen
        self.input_bits = input_bits
        self.output_bits = output_bits
        self.temper = CNNTemper(output_bits, output_bits)
        self.inverse_temper = CNNTemper2(input_bits)
        self.twister = CNNTwister(input_bits, output_bits, seqlen)
        self.activation = ACTIVATION
        # harden, binary_gumbel_softmax, torch.tanh, self.activation, torch.leaky_relu, etc
        self.sign_func = torch.tanh
        self.dropout = nn.Dropout1d(p=0.8)

    def autoencoder(self, x):
        x = x.reshape(-1, self.input_bits)
        register = self.inverse_temper(x)
        autoencoder = self.temper(self.sign_func(register))
        return autoencoder

    def forward(self, x):
        x = x.reshape(-1, self.seqlen, self.input_bits)
        x = x.transpose(1, 2)
        register = self.inverse_temper(x)
        register = register.transpose(1, 2)
        register = self.sign_func(register)
        register = self.dropout(register)

        twisted_register = self.twister(register)
        twisted_register = self.sign_func(twisted_register)
        output = self.temper(twisted_register.squeeze())
        return output

    def freeze(self):
        for param in self.inverse_temper.parameters():
            param.requires_grad = False
        for param in self.twister.parameters():
            param.requires_grad = False

    def load(self, temper_dir, inverse_temper_dir, twister_dir):
        if inverse_temper_dir is not None:
            self.inverse_temper.load_state_dict(torch.load(inverse_temper_dir))
            print("Loading inverse temper from ", inverse_temper_dir)
        if temper_dir is not None:
            self.temper.load_state_dict(torch.load(temper_dir))
            print("Loading temper from ", temper_dir)
        if twister_dir is not None:
            self.twister.load_state_dict(torch.load(twister_dir))
            print("Loading twister from ", twister_dir)

    def load_fully(self, model_dir):
        self.load_state_dict(torch.load(model_dir))


class ResFC(nn.Module):

    def __init__(self, output_bits=256, input_bits=8, seqlen=100):
        super(ResFC, self).__init__()
        MIDDLE_SHAPE = 256
        self.activation = ACTIVATION
        self.input_bits = input_bits
        self.seqlen = seqlen
        self.input_fc = nn.Linear(seqlen * input_bits, MIDDLE_SHAPE)
        self.residual = self.make_residual(MIDDLE_SHAPE, times=64)
        self.output_fc = nn.Linear(MIDDLE_SHAPE, output_bits)

    def make_residual(self, shape=256, times=5):
        layers = []
        for i in range(times):
            layers.append(ResBlock(shape, self.activation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.activation(self.input_fc(x))
        x = self.residual(x)
        x = self.output_fc(x)
        return x
