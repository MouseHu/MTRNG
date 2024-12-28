import torch

from .nalu import CustomRNN, NeuralAdder
from torch import nn
import torch.nn.functional as F
from torch_util import weight_init

# ALPHA = 0.5388
ALPHA = 0.5388
ACTIVATION = nn.LeakyReLU(negative_slope=ALPHA)


def pad_channels(in_channels, out_channels):
    def pad_func(x):
        x = x.transpose(-1, -2)
        x = F.interpolate(x, size=out_channels, mode='nearest')
        x = x.transpose(-1, -2)
        return x

    return pad_func


class ResBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, kernel_size=33, add_activation=True):
        super(ResBlock, self).__init__()
        self.operand_dim = kernel_size
        self.add_activation = add_activation
        self.conv1 = nn.Conv1d(in_channels, middle_channels, kernel_size)
        self.conv2 = nn.Conv1d(middle_channels, out_channels, kernel_size)

        self.bn1 = nn.BatchNorm1d(middle_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        if out_channels != in_channels:
            self.down_sample = pad_channels(in_channels, out_channels)
        else:
            self.down_sample = nn.Identity()
        if add_activation:
            self.activation = ACTIVATION
            alpha = ALPHA
        else:
            self.activation = nn.Identity()
            alpha = 0
        weight_init(self.conv1, alpha)
        weight_init(self.conv2, alpha)

    def forward(self, x):
        pad_x = F.pad(x, (self.operand_dim - 1, 0))
        residual = self.down_sample(x)
        feature = self.activation(self.bn1(self.conv1(pad_x)))
        feature = F.pad(feature, (self.operand_dim - 1, 0))
        output = self.bn2(self.conv2(feature)) + residual
        output = self.activation(output)
        # print(output.shape)
        return output


class CNNNeuralMultiplier(nn.Module):
    def __init__(self, operand_dim=16):
        super(CNNNeuralMultiplier, self).__init__()
        self.operand_dim = operand_dim
        self.cnn1 = ResBlock(in_channels=3, middle_channels=4, out_channels=8, kernel_size=operand_dim)
        self.cnn2 = ResBlock(in_channels=8, middle_channels=16, out_channels=32, kernel_size=operand_dim)
        self.cnn3 = ResBlock(in_channels=32, middle_channels=64, out_channels=128, kernel_size=operand_dim)
        self.cnn4 = ResBlock(in_channels=128, middle_channels=128, out_channels=16, kernel_size=operand_dim)
        self.adder = NeuralAdder(sum_dim=True, final_result=True)

    def forward(self, x):
        # batch_size, input_dim, length = x.shape
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        panel = self.cnn4(x)
        out = self.adder(panel)
        out = out.squeeze()  # batch_size * num_operand*x_dim
        return out


class LehmerCNNCracker(nn.Module):
    def __init__(self, seqlen=1, input_bits=128, output_bits=128):
        super(LehmerCNNCracker, self).__init__()
        self.out = CNNNeuralMultiplier(operand_dim=32)

    def forward(self, x):
        out = self.out(x)
        return out  # select high end bits
