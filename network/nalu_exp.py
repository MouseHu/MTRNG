import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from .nalu import CustomRNN


class ExpAdderCell(nn.Module):
    def __init__(self, input_shape=4, select_shape=4, output_shape=1, num_timesteps=64):
        super(ExpAdderCell, self).__init__()
        self.num_timesteps = 64
        self.W = Parameter(torch.rand(input_shape, select_shape, select_shape, num_timesteps))
        self.Wout = Parameter(torch.rand(select_shape, output_shape, num_timesteps))
        self.bout = Parameter(torch.rand(output_shape, num_timesteps))

    def forward(self, x, h, t):
        W_t = self.W[..., t]
        W_o = self.Wout[..., t]
        b_o = self.bout[..., t]
        # h = (batch_size, select_shape)
        # print(W_t.shape, h.shape)
        cb = (W_t @ h).mean(axis=0)
        ca = (x.transpose(0, 1) @ W_t).mean(axis=-1)
        bilinear = cb + ca
        # batch_size, select_shape, select_shape => batch_size, select_shape
        # print("adder cell", x.shape)
        # new_h = (x + h) / 2
        output = torch.cos(torch.pi * bilinear)
        new_h = 1. / 2 * (bilinear - output)
        output = output.transpose(0, 1) @ W_o + b_o
        return output, new_h


class ExpNeuralAdder(nn.Module):
    def __init__(self, input_shape=4, select_shape=4, output_shape=1, num_timesteps=64):
        super().__init__()
        self.cell = ExpAdderCell(input_shape, select_shape, output_shape, num_timesteps)
        self.rnn = CustomRNN(input_shape, select_shape, self.cell)

    def forward(self, input):
        # input shape: batch_size * num_bit * num_input
        # input = input.transpose(1, 2)
        output, hidden = self.rnn(input)
        return output


class ExpNeuralMultiplier(nn.Module):
    def __init__(self, input_dim=1, operand_num=1, operand_dim=64, input_length=64):
        super(ExpNeuralMultiplier, self).__init__()
        self.input_dim = input_dim
        self.operand_dim = operand_dim
        self.operand_num = operand_num
        operand_tensor = torch.concat([torch.ones(operand_dim, input_dim,1), torch.zeros(operand_dim, input_dim,1)],dim=-1)
        self.operand = Parameter(operand_tensor)
        # self.operand = Parameter(torch.rand(operand_num, input_dim, 2))
        self.adder = ExpNeuralAdder(input_shape=2, select_shape=2, output_shape=1,
                                    num_timesteps=input_length)

    def forward(self, x):
        # batch_size, input_dim, length = x.shape
        x = F.pad(x, (self.operand_dim - 1, 0))
        panel = F.conv1d(x, self.operand, stride=1)  # panel shape: batch_size * out_channel * x_length
        out = self.adder(panel)
        out = out.squeeze()  # batch_size * num_operand*x_dim
        return out


class LehmerExpCracker(nn.Module):
    def __init__(self, seqlen=1, input_bits=128, output_bits=128):
        super(LehmerExpCracker, self).__init__()
        self.out = ExpNeuralMultiplier(input_dim=3, operand_num=4, operand_dim=44, input_length=64)

    def forward(self, x):
        out = self.out(x)
        return out  # select high end bits
