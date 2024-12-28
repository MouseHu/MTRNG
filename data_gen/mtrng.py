#! coding: utf-8
import numpy as np
from numba import jit


@jit(nopython=True)
def gen_data(data_size):
    __n = 624
    __m = 397
    __a = 0x9908b0df
    __b = 0x9d2c5680
    __c = 0xefc60000
    __kInitOperand = 0x6c078965
    __kMaxBits = 0xffffffff
    __kUpperBits = 0x80000000
    __kLowerBits = 0x7fffffff
    __register = [0] * __n
    __state = 0

    for i in range(1, __n):
        prev = __register[i - 1]
        temp = __kInitOperand * (prev ^ (prev >> 30)) + i
        __register[i] = temp & __kMaxBits

    def __twister():
        for i in range(__n):
            y = (__register[i] & __kUpperBits) + \
                (__register[(i + 1) % __n] & __kLowerBits)
            __register[i] = __register[(i + __m) % __n] ^ (y >> 1)
            if y % 2:
                __register[i] ^= __a
        return None

    def __inverse_right_shift_xor(value, shift):
        i, result = 0, 0
        while i * shift < 32:
            part_mask = ((__kMaxBits << (32 - shift)) & __kMaxBits) >> (i * shift)
            part = value & part_mask
            value ^= part >> shift
            result |= part
            i += 1
        return result

    def __inverse_left_shift_xor(value, shift, mask):
        i, result = 0, 0
        while i * shift < 32:
            part_mask = (__kMaxBits >> (32 - shift)) << (i * shift)
            part = value & part_mask
            value ^= (part << shift) & mask
            result |= part
            i += 1
        return result

    def __inverse_temper(tempered):
        value = tempered
        value = __inverse_right_shift_xor(value, 18)
        value = __inverse_left_shift_xor(value, 15, __c)
        value = __inverse_left_shift_xor(value, 7, __b)
        value = __inverse_right_shift_xor(value, 11)
        return value

    # mt = MersenneTwister(0)
    # ti = TemperInverser()
    mt_data = np.zeros(data_size, dtype=np.uint32)
    register_data = np.zeros(data_size, dtype=np.uint32)
    # inverse_data = np.zeros(data_size, dtype=np.uint32)
    print("Generating Data")
    for i in range(data_size):
        if __state == 0:
            __twister()

        y = __register[__state]
        register_data[i] = y
        y = y ^ (y >> 11)
        y = y ^ (y << 7) & __b
        y = y ^ (y << 15) & __c
        y = y ^ (y >> 18)

        __state = (__state + 1) % __n
        mt_data[i] = y
        if i == 0:
            print("first dat", mt_data[0])
        assert register_data[i] == __inverse_temper(mt_data[i])
        if i % (2 ** 20) == 0:
            print(i)
    return mt_data, register_data


if __name__ == "__main__":
    log_size = 26
    print(log_size)
    data_size = 2 ** log_size
    mt_data, register_data = gen_data(data_size)
    print("Dumping Data")
    mt_data.tofile("../data/mtrng_{}.dat".format(log_size))
    register_data.tofile("../data/register_mtrng_{}.dat".format(log_size))
