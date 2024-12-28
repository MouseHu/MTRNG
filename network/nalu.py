import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch_util import binary_gumbel_softmax, harden, weight_init
import numpy as np


class AdderCell(nn.Module):
    def __init__(self, sum_dim=True, final_result=True, require_grad=True):
        super(AdderCell, self).__init__()
        self.sum_dim = sum_dim
        self.w1 = Parameter(-1 * torch.ones(1) / 2, requires_grad=require_grad)
        self.b1 = Parameter(torch.ones(1) / 2, requires_grad=require_grad)
        if final_result:
            self.w2 = Parameter(2 * torch.ones(1), requires_grad=require_grad)
            self.b2 = Parameter(-1 * torch.ones(1), requires_grad=require_grad)
        else:
            self.w2 = Parameter(torch.ones(1), requires_grad=require_grad)
            self.b2 = Parameter(torch.zeros(1), requires_grad=require_grad)
        self.w3 = Parameter(torch.ones(1) / 2, requires_grad=require_grad)

    def forward(self, x, h, t):  # (current_input, hidden_input, time_step)
        if self.sum_dim:
            x = x.sum(-2, keepdims=True)  # -1 is the batch_dim
            h = h.sum(-2, keepdims=True)
        # print("adder cell", x.shape)
        # new_h = (x + h) / 2
        output = self.w1 * torch.cos(torch.pi * (x + h)) + self.b1
        new_h = self.w3 * ((x + h) - output)
        output = self.w2 * output + self.b2
        # print(output.shape)
        return output, new_h


class CustomRNN(nn.Module):

    def __init__(self, input_size, hidden_size, cell, batch_first=True):
        """Initialize params."""
        super(CustomRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.batch_first = batch_first
        self.cell = cell

    def forward(self, input, hidden=None):
        """Propogate input through the network."""
        if self.batch_first:
            input = input.transpose(0, -1)
        # batch_size = input.size(1)
        input_shape = input.shape[1:]
        if hidden is None:
            hidden = torch.zeros(*input_shape,
                                 dtype=input.dtype, device=input.device)

        output = []
        steps = range(input.size(0))
        for i in steps:
            out, hidden = self.cell(input[i], hidden, i)
            output.append(out)

            # output.append(hidden[0] if isinstance(hidden, tuple) else hidden)
            # output.append(isinstance(hidden, tuple) and hidden[0] or hidden)

        output = torch.cat(output, 0).view(input.size(0), *output[0].size())

        if self.batch_first:
            output = output.transpose(0, -1)

        return output, hidden


class NeuralAdder(nn.Module):
    def __init__(self, sum_dim=True, final_result=True):
        super().__init__()
        self.cell = AdderCell(sum_dim, final_result)
        self.rnn = CustomRNN(1, 1, self.cell)

    def forward(self, input):
        # input shape: batch_size * num_bit * num_input
        # input = input.transpose(1, 2)
        output, hidden = self.rnn(input)
        return output


class NeuralMultiAdder(nn.Module):
    def __init__(self, shift=0, final_result=False):
        super().__init__()
        self.shift = shift
        self.cell = AdderCell(final_result)
        self.rnn = CustomRNN(1, 1, self.cell)

    def forward(self, *input):
        input_tensor = torch.stack([*input], dim=1)  # num_input * batch_size * num_bit
        # move the first axis to the last
        # permutes = [i for i in range(1, len(input_tensor.shape))]
        # permutes.append(0)
        # input_tensor = torch.permute(input_tensor, permutes)
        # print(input_tensor.shape)
        input_tensor = torch.sum(input_tensor, dim=1, keepdim=True)
        output, hidden = self.rnn(input_tensor)
        # print(output.shape)
        if self.shift > 0:
            output = output[:, self.shift:]
        return output.squeeze()


class Rounder(nn.Module):
    def __init__(self, input_dim=128, shift=64):
        super(Rounder, self).__init__()
        self.input_dim = input_dim
        self.shift = shift
        self.adder = NeuralAdder(sum_dim=False, final_result=False)
        mask = np.zeros(input_dim)
        mask[0] = 1
        self.mask = Parameter(torch.from_numpy(mask).float(), requires_grad=True)

    def forward(self, x):
        # print(x.shape, self.mask.shape)
        integer = x[..., self.shift:]
        cond = x[..., self.shift - 1:self.shift] * self.mask
        output = self.adder(integer + cond)
        # if len(output.shape) > 2:
        #     output = output.transpose(1, 2)
        return output


class NeuralMultiplier(nn.Module):
    def __init__(self, operand_dim=64, shift=0, oracle=None, final_result=False):
        super(NeuralMultiplier, self).__init__()
        self.operand_dim = operand_dim
        self.shift = shift
        if oracle is not None:
            self.no_oracle = False
            self.operand = Parameter(torch.from_numpy(oracle).float())
            self.operand_dim = oracle.shape[0]
        else:
            self.no_oracle = True
            self.operand = Parameter(torch.randn(operand_dim))
            # self.operand = Parameter(torch.randn(operand_dim, 2))
            # self.output_operand = Parameter(torch.from_numpy(np.array([0, 1])).float())

        self.adder = NeuralAdder(final_result)

    def forward(self, x):
        if self.no_oracle:
            # operand = torch.sigmoid(self.operand)
            # operand = F.gumbel_softmax(self.operand, tau=1, hard=True)
            # operand = torch.matmul(operand, self.output_operand)

            # operand = operand[:,0]
            operand = self.operand
        else:
            operand = self.operand
        operand = operand.reshape(1, 1, self.operand_dim)
        x = F.pad(x, (self.operand_dim - 1, 0))
        # print(x.shape)
        batch_size, padded_length = x.shape
        x = x.reshape(batch_size, 1, padded_length)
        panel = F.conv1d(x, operand, stride=1)
        # panel = panel.reshape(batch_size, self.operand_dim, 1)
        out = self.adder(panel)
        out = out.squeeze()  # batch_size * x_dim
        if self.shift > 0:
            out = out[:, self.shift:]
            out = F.pad(out, (0, self.shift))

        return out


class LehmerCracker(nn.Module):
    def __init__(self):
        super(LehmerCracker, self).__init__()
        # r1, r2, r3, s1, s2, s3, t1, t2, t3 = np.array(
        #     [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0,
        #       0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1],
        #      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
        #       0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1],
        #      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0,
        #       0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0],
        #      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0,
        #       0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0],
        #      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1,
        #       1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        #      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0,
        #       0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1],
        #      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0,
        #       1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
        #      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0,
        #       0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0],
        #      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0,
        #       1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0]])
        # ur, us, ut, vr, vs, vt, a = np.array(
        #     [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #       0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0],
        #      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #       0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0],
        #      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #       0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1],
        #      [0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0,
        #       1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0],
        #      [0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1,
        #       1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0],
        #      [0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1,
        #       0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        #      [1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1,
        #       0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1]])
        # a3 = np.array(
        #     [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1,
        #      0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1,
        #      1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,
        #      0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1])
        self.adder = NeuralMultiAdder()
        self.rounder = Rounder(shift=64)
        # self.r1, self.r2, self.r3 = NeuralMultiplier(oracle=r1), NeuralMultiplier(oracle=r2), NeuralMultiplier(
        #     oracle=r3)
        # self.s1, self.s2, self.s3 = NeuralMultiplier(oracle=s1), NeuralMultiplier(oracle=s2), NeuralMultiplier(
        #     oracle=s3)
        # self.t1, self.t2, self.t3 = NeuralMultiplier(oracle=t1), NeuralMultiplier(oracle=t2), NeuralMultiplier(
        #     oracle=t3)
        # self.ur, self.us, self.ut = NeuralMultiplier(oracle=ur), NeuralMultiplier(oracle=us), NeuralMultiplier(
        #     oracle=ut)
        # self.vr, self.vs, self.vt = NeuralMultiplier(oracle=vr), NeuralMultiplier(oracle=vs), NeuralMultiplier(
        #     oracle=vt)
        # self.au = NeuralMultiplier(oracle=a)
        # self.out = NeuralMultiplier(oracle=a3, final_result=True)
        self.r1, self.r2, self.r3 = NeuralMultiplier(operand_dim=128), NeuralMultiplier(
            operand_dim=128), NeuralMultiplier(operand_dim=128)
        self.s1, self.s2, self.s3 = NeuralMultiplier(operand_dim=128), NeuralMultiplier(
            operand_dim=128), NeuralMultiplier(operand_dim=128)
        self.t1, self.t2, self.t3 = NeuralMultiplier(operand_dim=128), NeuralMultiplier(
            operand_dim=128), NeuralMultiplier(operand_dim=128)
        self.ur, self.us, self.ut = NeuralMultiplier(operand_dim=128), NeuralMultiplier(
            operand_dim=128), NeuralMultiplier(operand_dim=128)
        self.vr, self.vs, self.vt = NeuralMultiplier(operand_dim=128), NeuralMultiplier(
            operand_dim=128), NeuralMultiplier(operand_dim=128)
        self.au = NeuralMultiplier(operand_dim=128)
        self.out = NeuralMultiplier(operand_dim=128, final_result=True)

    def forward(self, x):
        x = x.transpose(1, 2)
        x1, x2, x3 = x[:, :, 0], x[:, :, 1], x[:, :, 2]
        x1, x2, x3 = F.pad(x1, (0, 128)), F.pad(x2, (0, 128)), F.pad(x3, (0, 128))
        # print(x1.shape, x2.shape, x3.shape)
        r1, r2, r3 = self.r1(x1), self.r2(x2), self.r3(x3)
        s1, s2, s3 = self.s1(x1), self.s2(x2), self.s3(x3)
        t1, t2, t3 = self.t1(x1), self.t2(x2), self.t3(x3)
        # print("r123", r1.shape, r2.shape, r3.shape)
        r = self.adder(r1, r2, r3)
        s = self.adder(s1, s2, s3)
        t = self.adder(t1, t2, t3)
        # print(r.shape, s.shape, t.shape)
        # print("rst", r.shape)
        r, s, t = self.rounder(r), self.rounder(s), self.rounder(t)
        ur, us, ut = self.ur(r), self.us(s), self.ut(t)
        vr, vs, vt = self.vr(r), self.vs(s), self.vt(t)
        u = self.adder(ur, us, ut)
        v = self.adder(vr, vs, vt)
        au = self.au(u)
        state = self.adder(au, v)
        out = self.out(state)
        out = out[:, 64:]
        return out
