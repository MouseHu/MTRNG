from lehmer64 import lehmer64
import numpy as np

lhemer64_generator, state_printer = lehmer64()


def get_binary(x):
    binary_output = np.unpackbits(x.view(np.uint8), bitorder='little').reshape(*x.shape, -1)
    binary_output = binary_output[..., ::-1].squeeze()
    return binary_output


# start = 35
x = [lhemer64_generator() for _ in range(1000)]
x = np.array(x, dtype=np.uint64)
floatx = np.array(x, dtype=np.float128)
# floatx = np.array(x, dtype=np.double)
logx = np.log2(floatx)
expx = np.exp2(logx).astype(np.uint64)
binary_output = get_binary(expx)
binary_label = get_binary(x)
# expx = np.exp2(logx)

print(np.mean(binary_output != binary_label))
