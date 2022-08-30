import numpy as np


def get_bin(x):
    if abs(x) < 1:
        x = round(x * (2 ** 64))
    if x < 0:
        x = 2 ** 192 + x
    binary_x = bin(x)
    binary_x = [int(a) for a in binary_x[2:]]
    if len(binary_x) <= 64:
        max_len = 64
    elif len(binary_x) <= 128:
        max_len = 128
    # else:
    # max_len = 64
    while len(binary_x) < max_len:
        binary_x.insert(0, 0)
    print(binary_x)
    return binary_x


data = [2.64929081169728e-7, 3.51729342107376e-7, 3.89110109147656e-8, 3.12752538137199e-7, -1.00664345453760e-7,
        -2.16685184476959e-7, 3.54263598631140e-8, -2.05535734808162e-7, 2.73269247090513e-7]

data2 = [1556524, 2249380, 1561981,
        8429177212358078682, 4111469003616164778, 3562247178301810180, 0xda942042e4dd58b5,
        0xdb76c43996e558d0bdfbbe1277f2430d]

out = [get_bin(a) for a in data2]

print(out)

# 5899963497747922713
