import torch
from torch import nn
import torch.nn.functional as F
from torch_util import binary_gumbel_softmax, harden, weight_init

# ALPHA = 0.5388
ALPHA = 0.5388
NUM_FEATURES = 1024
NUM_FEATURES_TWISTER = 128
ACTIVATION = nn.LeakyReLU(negative_slope=ALPHA)


class ResBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, kernel_size=33, add_activation=True):
        super(ResBlock, self).__init__()
        self.add_activation = add_activation
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(in_channels, middle_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(middle_channels, track_running_stats=False)
        self.conv2 = nn.Conv1d(middle_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels, track_running_stats=False)
        if out_channels != in_channels:
            self.down_sample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1),
                nn.BatchNorm1d(out_channels, track_running_stats=False)
            )
        else:
            self.down_sample = nn.Identity()
        self.activation = ACTIVATION

        weight_init(self.conv1, ALPHA)
        weight_init(self.conv2, ALPHA)

    def forward(self, x):
        residual = self.down_sample(x)
        feature = self.activation(self.bn1(self.conv1(x)))
        output = self.bn2(self.conv2(feature)) + residual
        if self.add_activation:
            output = self.activation(output)
        return output


class CNNTemper(nn.Module):
    def __init__(self, input_bits=32, output_bits=32, kernel_size=19):
        super().__init__()
        assert input_bits == output_bits
        self.input_bits = input_bits
        # kernel_size = 33
        self.res_cnn1 = ResBlock(in_channels=1, middle_channels=8, out_channels=16, kernel_size=kernel_size)
        self.res_cnn2 = ResBlock(in_channels=16, middle_channels=32, out_channels=1, kernel_size=kernel_size)
        # self.res_cnn1 = ResBlock(in_channels=input_bits, middle_channels=32, out_channels=32, kernel_size=1)
        # self.res_cnn2 = ResBlock(in_channels=32, middle_channels=32, out_channels=input_bits, kernel_size=1)
        # self.cnn1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=33, padding=16)
        # self.cnn2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=33, padding=16)
        # self.cnn3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=33, padding=16)
        # self.cnn4 = nn.Conv1d(in_channels=32, out_channels=1, kernel_size=33, padding=16)
        # self.bn1 = nn.BatchNorm1d(num_features=8)
        # self.bn2 = nn.BatchNorm1d(num_features=16)
        # self.bn3 = nn.BatchNorm1d(num_features=32)
        # self.bn4 = nn.BatchNorm1d(num_features=1)
        # self.fc1 = nn.Linear(input_bits, output_bits)
        self.activation = ACTIVATION

    def forward(self, x):
        x = x.reshape(-1, 1, self.input_bits)
        # feature = self.activation(self.bn1(self.cnn1(x)))
        # feature = self.activation(self.bn2(self.cnn2(feature)))
        # feature = self.activation(self.bn3(self.cnn3(feature)))
        # feature = self.cnn4(feature)
        feature = self.res_cnn1(x)
        feature = self.res_cnn2(feature)
        output = feature.squeeze()
        return output


class CNNTemper2(nn.Module):
    def __init__(self, input_bits=32):
        super().__init__()
        self.input_bits = input_bits
        # kernel_size = 33
        self.res_cnn1 = ResBlock(in_channels=input_bits, middle_channels=32, out_channels=32, kernel_size=1)
        self.res_cnn2 = ResBlock(in_channels=32, middle_channels=32, out_channels=input_bits, kernel_size=1)
        self.activation = ACTIVATION

    def forward(self, x):
        # x = x.reshape(-1, 1, self.input_bits)
        feature = self.res_cnn1(x)
        feature = self.res_cnn2(feature)
        output = feature.squeeze()
        return output


class ResCNNTemper(nn.Module):
    def __init__(self, input_bits=32, output_bits=32, kernel_size=7):
        super().__init__()
        self.input_bits = input_bits
        self.res_cnn1 = ResBlock(in_channels=1, middle_channels=4, out_channels=4, kernel_size=kernel_size)
        self.res_cnn2 = ResBlock(in_channels=4, middle_channels=8, out_channels=8, kernel_size=kernel_size)
        self.res_cnn3 = ResBlock(in_channels=8, middle_channels=16, out_channels=16, kernel_size=kernel_size)
        self.res_cnn4 = ResBlock(in_channels=16, middle_channels=32, out_channels=32, kernel_size=kernel_size)
        self.res_cnn5 = ResBlock(in_channels=32, middle_channels=64, out_channels=64, kernel_size=kernel_size)
        self.res_cnn6 = ResBlock(in_channels=64, middle_channels=64, out_channels=1, kernel_size=kernel_size,
                                 add_activation=False)

    def forward(self, x):
        x = x.reshape(-1, 1, self.input_bits)
        feature = self.res_cnn1(x)
        feature = self.res_cnn2(feature)
        feature = self.res_cnn3(feature)
        feature = self.res_cnn4(feature)
        feature = self.res_cnn5(feature)
        feature = self.res_cnn6(feature)
        output = feature.squeeze()
        return output





class CNNTwister(nn.Module):
    def __init__(self, input_bits=32, output_bits=32, seqlen=624):
        super().__init__()
        self.seqlen = seqlen
        self.input_bits = input_bits
        # self.res_cnn1 = ResBlock(in_channels=seqlen, middle_channels=256, out_channels=128, kernel_size=33)
        # self.res_cnn2 = ResBlock(in_channels=128, middle_channels=64, out_channels=32, kernel_size=33)
        # self.res_cnn3 = ResBlock(in_channels=32, middle_channels=16, out_channels=8, kernel_size=33)
        self.cnn = nn.Conv1d(in_channels=seqlen, out_channels=4, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(4, NUM_FEATURES_TWISTER)
        self.fc2 = nn.Linear(NUM_FEATURES_TWISTER, 1)
        self.bn1 = nn.BatchNorm1d(input_bits, track_running_stats=False)
        # self.activation = ACTIVATION
        self.activation = nn.LeakyReLU(negative_slope=0.2)
        weight_init(self.fc1, ALPHA)
        weight_init(self.fc2, ALPHA)

    def forward(self, x):
        x = x.reshape(-1, self.seqlen, self.input_bits)
        cnn_out = self.activation(self.cnn(x))
        # cnn_out = self.res_cnn1(x)
        # cnn_out = self.res_cnn2(cnn_out)
        # cnn_out = self.res_cnn3(cnn_out)
        cnn_out = cnn_out.transpose(1, 2)
        feature = self.activation(self.bn1(self.fc1(cnn_out)))
        out = self.fc2(feature)

        return out
