import torch
from torch import nn
import torch.nn.functional as F
from network.cnn import ResBlock
ALPHA = 0.5388
MIDDLE_SHAPE = 1024
NUM_FEATURES = 1024
NUM_MIDDLE = 1024


def weight_init(layer, alpha=0.01):
    nn.init.kaiming_uniform_(layer.weight, mode='fan_in', a=alpha, nonlinearity='leaky_relu')
    nn.init.zeros_(layer.bias)


def weight_init_batchnorm(layer):
    nn.init.constant_(layer.weight, 1)
    nn.init.zeros_(layer.bias)


class ResFCBlock(nn.Module):
    def __init__(self, shape=256, activation=F.leaky_relu):
        super(ResFCBlock, self).__init__()
        self.block1 = nn.Linear(shape, NUM_MIDDLE)
        self.block2 = nn.Linear(NUM_MIDDLE, shape)
        self.activation = activation
        self.bn1 = nn.BatchNorm1d(num_features=NUM_MIDDLE)
        self.bn2 = nn.BatchNorm1d(num_features=shape)
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)
        weight_init(self.block1, alpha=ALPHA)
        weight_init_batchnorm(self.bn1)
        weight_init_batchnorm(self.bn2)
        weight_init(self.block2, alpha=ALPHA)

    def forward(self, x):
        y = self.block1(x)
        y = self.bn1(y)
        y = self.activation(y)
        # y = self.dropout1(y)
        y = self.block2(y)
        y = self.bn2(y)
        y = y + x
        y = self.activation(y)
        return y


class TemperInverser(nn.Module):

    def __init__(self, output_bits=32, input_bits=32, num_layers=2, name="inversor"):
        super(TemperInverser, self).__init__()
        self.output_bits = output_bits
        self.activation = F.elu
        self.input_bits = input_bits
        self.residual = self.make_residual(input_bits, num_layers, name=name)

    def make_residual(self, shape=256, times=5, name="residual"):
        layers = []
        for i in range(times):
            layers.append(ResFCBlock(shape, activation=self.activation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.residual(x)
        return x


class MTCracker(nn.Module):
    def __init__(self, input_bits=32, seqlen=3):
        super(MTCracker, self).__init__()
        self.input_bits = input_bits
        self.output_bits = input_bits
        self.seqlen = seqlen
        self.activation = F.elu

        num_features = 32
        self.rotator1 = ResBlock(in_channels=seqlen, middle_channels=1024, out_channels=4, kernel_size=3)
        self.rotator2 = nn.Linear(4, num_features)
        self.rotator3 = nn.Linear(num_features, 1)

        self.inversor = TemperInverser(self.output_bits, input_bits, num_layers=1)
        self.inversor2 = TemperInverser(self.output_bits, input_bits, num_layers=1)

        self.dropout = nn.Dropout1d(p=0.15)
        # weight_init(self.rotator1, alpha=ALPHA)
        weight_init(self.rotator2, alpha=ALPHA)
        weight_init(self.rotator3, alpha=ALPHA)

    def forward(self, x):
        flat_x = x.reshape(-1, self.input_bits)
        register = self.inversor(flat_x)
        register = self.activation(register)

        register = register.reshape(-1, self.seqlen, self.input_bits)
        register = self.dropout(register)

        rotated = self.activation(self.rotator1(register))
        rotated = torch.transpose(rotated, 1, 2)
        rotated = self.activation(self.rotator2(rotated))
        rotated = self.activation(self.rotator3(rotated))
        rotated = torch.transpose(rotated, 1, 2).reshape(-1, self.input_bits)

        out = self.inversor2(rotated)
        return out





