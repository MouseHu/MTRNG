import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


def get_binary_exp(logs):
    exp = torch.exp2(logs)
    binary_output = (exp.detach().cpu().numpy()).astype(np.uint64)
    # print(binary_output)
    binary_output = np.unpackbits(binary_output.view(np.uint8), bitorder='little').reshape(*binary_output.shape, -1)
    binary_output = binary_output[..., ::-1].squeeze()
    return binary_output


def mse_loss(x, y, binary_labels=None):
    info = {}
    loss = ((x - y) ** 2).mean()
    if binary_labels is None:
        correct = loss
    else:
        binary_labels = binary_labels.detach().cpu().numpy()
        binary_output = get_binary_exp(x)
        # binary_labels2 = get_binary_exp(y)
        correct = np.mean(binary_output == binary_labels)
        # problem = np.mean(binary_labels2[..., :40] == binary_labels[..., :40])
        # problem = np.mean(binary_labels == binary_labels2)
        # correct = 1
        # print(correct, problem)
    return loss, correct, info


def mtrng_loss(model, x, y, l1_coef=0, predict_list=None):
    if predict_list is None:
        predict_list = np.arange(y.shape[1])
    batch_size = y.shape[0]
    x, y = x.squeeze(), y.float().squeeze()
    x, y = x[:, predict_list], y[:, predict_list]
    x, y = x.reshape(-1), y.reshape(-1)
    prob = torch.sigmoid(x).reshape(-1)
    loss = nn.BCEWithLogitsLoss()(x, y)
    # loss = slope_loss(x, y)

    l1_loss = 0.
    for param in model.parameters():
        l1_loss += param.abs().sum()

    loss = loss + l1_coef * l1_loss
    predict = prob > 0.5
    correct = (predict == y).sum() / len(predict_list) / batch_size
    info = {"l1_loss": l1_loss, "batch_size": batch_size}

    return loss, correct, info


def l1_norm(model):
    l1_regularization = 0.
    for param in model.parameters():
        l1_regularization += param.abs().sum()
    return l1_regularization


def slope_loss(x, y):
    y = y * 2 - 1.
    sign = torch.tanh(x) * y
    loss = 1 - F.leaky_relu(sign, negative_slope=0.01)
    return loss.mean()


def binary_gumbel_softmax(logits):
    scale = 10
    prob = torch.sigmoid(logits) * scale
    gumbel_logits = torch.stack([prob, scale - prob], dim=-1)
    gumbel = F.gumbel_softmax(gumbel_logits, tau=1, hard=True)
    gumbel = gumbel[..., 0]
    gumbel = gumbel * 2 - 1.
    return gumbel


def harden(logits):
    soft_sign = torch.tanh(logits)
    hard_sign = torch.sign(logits)
    output = hard_sign.detach() + soft_sign - soft_sign.detach()
    return output


def binary_torch(x, bits=32):
    mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()


def binary(x):
    np.unpackbits(x.view(np.uint8))


def weight_init(layer, alpha=0.01):
    # nn.init.kaiming_uniform_(layer.weight, mode='fan_in', a=alpha, nonlinearity='leaky_relu')
    nn.init.kaiming_normal_(layer.weight, mode='fan_in', a=alpha, nonlinearity='leaky_relu')


def l1_difference(model1, model2):
    return sum((x - y).abs().sum() for x, y in zip(model1.parameters(), model2.parameters()))


def load_ddp_model(model_dir, model):
    state_dict = torch.load(model_dir)

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove 'module.' of DataParallel/DistributedDataParallel
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    print("load ddp successfully")
