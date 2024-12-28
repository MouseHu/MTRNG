import torch
from torch import nn
import math

# ALPHA = 0.5388
ALPHA = 0.5388
NUM_FEATURES = 2048
NUM_FEATURES_TWISTER = 128
# ACTIVATION = nn.LeakyReLU(negative_slope=ALPHA)
ACTIVATION = nn.ELU()


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=128):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


class Multiplier(nn.Module):
    def __init__(self, input_bits=128, output_bits=128, seqlen=128):
        super().__init__()
        self.input_bits = input_bits
        self.output_bits = output_bits

        self.rnn = nn.LSTM(input_size=16, hidden_size=256, num_layers=3, batch_first=True, bidirectional=True)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(256 * 2, 1)
        self.fc1 = nn.Linear(1, 16)
        self.activation = ACTIVATION
        self.position_encoding = PositionalEncoding(16, input_bits)

    def forward(self, x):
        batch_size, input_bits = x.shape
        x = x.reshape(batch_size, input_bits, 1)  # (batch_size, input_bits, 1)
        x = self.activation(self.bn1(self.fc1(x)))
        x = self.position_encoding(x)
        middle, h_n = self.rnn(x)  # (batch_size,  input_bits, hidden_size)
        output = self.fc2(middle)  # (batch_size,  input_bits, 1)
        output = output.squeeze()
        output = output[:, :self.output_bits]  # (batch_size, output_bits)
        return output
