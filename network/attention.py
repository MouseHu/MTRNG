import torch
from torch import nn

# import matplotlib.pyplot as plt

ALPHA = 0.5388
NUM_FEATURES = 256


def weight_init(layer, alpha=0.01):
    nn.init.kaiming_uniform_(layer.weight, mode='fan_in', a=alpha, nonlinearity='leaky_relu')


class AttentionLayer(nn.Module):
    def __init__(self, input_dim=32, embed_dim=32, output_dim=32, seqlen=624):
        super(AttentionLayer, self).__init__()
        self.seqlen = seqlen
        self.input_dim = input_dim
        self.W_Q = nn.Linear(input_dim, embed_dim)
        self.W_K = nn.Linear(input_dim, embed_dim)
        self.W_V = nn.Linear(input_dim, output_dim)
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads=1)

        weight_init(self.W_Q, ALPHA)
        weight_init(self.W_K, ALPHA)
        weight_init(self.W_V, ALPHA)

    def forward(self, x):
        x = x.reshape(-1, self.seqlen, self.input_dim)
        x = x.transpose(0, 1)
        query = self.W_Q(x)
        key = self.W_K(x)
        value = self.W_V(x)
        attn_output, attn_output_weights = self.multihead_attn(query, key, value)
        attn_output = attn_output.transpose(0, 1)
        return attn_output


class EncoderBlock(nn.Module):

    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0):
        """
        Inputs:
            input_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            dim_feedforward - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()

        # Attention layer
        self.self_attn = AttentionLayer(input_dim, input_dim, num_heads)

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim)
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Attention part
        attn_out = self.self_attn(x, mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # MLP part
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)

        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
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


class TransformerEncoder(nn.Module):

    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(**block_args) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for l in self.layers:
            x = l(x, mask=mask)
        return x

    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for l in self.layers:
            _, attn_map = l.self_attn(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = l(x)
        return attention_maps


class AttentionTwister(nn.Module):
    def __init__(self, input_bits=32, output_bits=32, seqlen=624):
        super().__init__()
        self.attention1 = AttentionLayer(input_dim=input_bits, embed_dim=seqlen, output_dim=output_bits,
                                        seqlen=seqlen)
        self.attention2 = AttentionLayer(input_dim=input_bits, embed_dim=seqlen, output_dim=output_bits,
                                         seqlen=seqlen)
        self.attention3 = AttentionLayer(input_dim=input_bits, embed_dim=seqlen, output_dim=output_bits,
                                         seqlen=seqlen)
        self.fc1 = nn.Linear(seqlen, NUM_FEATURES)
        self.fc2 = nn.Linear(NUM_FEATURES, output_bits)
        self.activation = nn.LeakyReLU(negative_slope=ALPHA)

        self.bn1 = nn.BatchNorm1d(NUM_FEATURES)
        # init
        weight_init(self.fc1, alpha=ALPHA)
        weight_init(self.fc2, alpha=ALPHA)

    def forward(self, x):
        attn_feature = self.attention1(x)
        attn_feature = attn_feature.mean(axis=1)
        attn_feature = self.attention2(attn_feature)
        attn_feature = attn_feature.mean(axis=1)
        attn_feature = self.attention3(attn_feature)
        attn_feature = attn_feature.mean(axis=1)
        feature = self.activation(self.bn1(self.fc1(attn_feature)))
        # feature = self.res1(feature)
        output = self.fc2(feature)
        return output

# if __name__ == "__main__":
#     print(torch.__version__)
#     x = torch.randn((2, 624, 32))
#     attention_layer = AttentionLayer()
#     y = attention_layer(x)
#     print(y.shape)
