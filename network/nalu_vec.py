import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import numpy as np
from network.nalu import CustomRNN, NeuralAdder, Rounder


class NeuralVecMultiplier(nn.Module):
    def __init__(self, input_dim=1, operand_num=1, operand_dim=64, oracle=None, final_result=False):
        super(NeuralVecMultiplier, self).__init__()
        self.input_dim = input_dim
        self.operand_dim = operand_dim
        self.operand_num = operand_num
        if oracle is not None:
            self.no_oracle = False
            self.operand = Parameter(torch.from_numpy(oracle).float())
            self.operand_dim = oracle.shape[0]
            self.operand_num = oracle.shape[1]
            self.input_dim = oracle.shape[2]
        else:
            self.no_oracle = True
            self.operand = Parameter(torch.rand(operand_num, input_dim, operand_dim))
            self.adder = NeuralAdder(final_result)

    def forward(self, x):
        # if self.no_oracle:
        #     operand = torch.sigmoid(self.operand)
        # else:
        # print(x.shape)
        operand = self.operand
        operand = operand.reshape(self.operand_num, self.input_dim, self.operand_dim)
        # batch_size, input_dim, length = x.shape
        x = F.pad(x, (self.operand_dim - 1, 0))
        # padded_length = length + self.operand_dim - 1
        # x = x.reshape(batch_size, input_dim, padded_length)
        panel = F.conv1d(x, operand, stride=1)  # panel shape: batch_size * out_channel * x_length
        # panel = panel.transpose(1, 2)  # panel shape: batch_size * x_length * out_channel
        out = self.adder(panel)
        # out = out.reshape(batch_size, self.operand_num, -1)
        out = out.squeeze()  # batch_size * num_operand*x_dim
        return out


class MultiplierCracker(nn.Module):
    def __init__(self, operand_num=64, operand_dim=64):
        super(MultiplierCracker, self).__init__()
        self.multiplier = NeuralVecMultiplier(input_dim=1, operand_num=operand_num, operand_dim=operand_dim,
                                              final_result=True)

        self.fc = nn.Linear(operand_num, 1)

    def forward(self, x):
        # x = x.transpose(0, 1)
        batch_size, length = x.shape
        x = x.reshape(batch_size, 1, length)
        x = self.multiplier(x)
        # x = F.relu(x)
        x = x.transpose(1, 2)
        x = self.fc(x)
        return x.squeeze()


class LehmerLearnedCracker(nn.Module):
    def __init__(self):
        super(LehmerLearnedCracker, self).__init__()
        self.rst = NeuralVecMultiplier(input_dim=3, operand_num=127, operand_dim=64)
        self.uv = NeuralVecMultiplier(input_dim=127, operand_num=128, operand_dim=64)
        self.out = NeuralVecMultiplier(input_dim=128, operand_num=1, operand_dim=64, final_result=True)
        self.rounder = Rounder(shift=64)

    def forward(self, x):
        x = F.pad(x, (0, 128))
        rst = self.rst(x)
        # rounded_rst = self.rounder(rst)
        rounded_rst = rst[..., 64:]
        uv = self.uv(rounded_rst)
        out = self.out(uv)
        out = out[..., 64:]  # select high end bits
        return out
