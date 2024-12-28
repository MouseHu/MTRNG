import sys
import math
import struct
import random
from z3 import *

MASK = 0xFFFFFFFFFFFFFFFF


# xor_shift_128_plus algorithm
def xs128p(state0, state1, browser):
    s1 = state0 & MASK
    s0 = state1 & MASK
    s1 ^= (s1 << 23) & MASK
    s1 ^= (s1 >> 17) & MASK
    s1 ^= s0 & MASK
    s1 ^= (s0 >> 26) & MASK
    state0 = state1 & MASK
    state1 = s1 & MASK
    if browser == 'chrome':
        generated = state0 & MASK
    else:
        generated = (state0 + state1) & MASK

    return state0, state1, generated


# Symbolic execution of xs128p
def sym_xs128p(slvr, sym_state0, sym_state1, generated, browser):
    s1 = sym_state0
    s0 = sym_state1
    s1 ^= (s1 << 23)
    s1 ^= LShR(s1, 17)
    s1 ^= s0
    s1 ^= LShR(s0, 26)
    sym_state0 = sym_state1
    sym_state1 = s1
    if browser == 'chrome':
        calc = sym_state0
    else:
        calc = (sym_state0 + sym_state1)

    condition = Bool('c%d' % int(generated * random.random()))
    if browser == 'chrome':
        impl = Implies(condition, LShR(calc, 12) == int(generated))
    elif browser == 'firefox' or browser == 'safari':
        # Firefox and Safari save an extra bit
        impl = Implies(condition, (calc & 0x1FFFFFFFFFFFFF) == int(generated))

    slvr.add(impl)
    return sym_state0, sym_state1, [condition]


def reverse17(val):
    return val ^ (val >> 17) ^ (val >> 34) ^ (val >> 51)


def reverse23(val):
    return (val ^ (val << 23) ^ (val << 46)) & MASK


def xs128p_backward(state0, state1, browser):
    prev_state1 = state0
    prev_state0 = state1 ^ (state0 >> 26)
    prev_state0 = prev_state0 ^ state0
    prev_state0 = reverse17(prev_state0)
    prev_state0 = reverse23(prev_state0)
    # this is only called from an if chrome
    # but let's be safe in case someone copies it out
    if browser == 'chrome':
        generated = prev_state0
    else:
        generated = (prev_state0 + prev_state1) & MASK
    return prev_state0, prev_state1, generated


# Firefox nextDouble():
# (rand_uint64 & ((1 << 53) - 1)) / (1 << 53)
# Chrome nextDouble():
# (state0 | 0x3FF0000000000000) - 1.0
# Safari weakRandom.get():
# (rand_uint64 & ((1 << 53) - 1) * (1.0 / (1 << 53)))
def to_double(browser, out):
    if browser == 'chrome':
        double_bits = (out >> 12) | 0x3FF0000000000000
        double = struct.unpack('d', struct.pack('<Q', double_bits))[0] - 1
    elif browser == 'firefox':
        double = float(out & 0x1FFFFFFFFFFFFF) / (0x1 << 53)
    elif browser == 'safari':
        double = float(out & 0x1FFFFFFFFFFFFF) * (1.0 / (0x1 << 53))
    return double


def crack(dubs, generate_len=10, browser='chrome'):
    if browser == 'chrome':
        dubs = dubs[:-1][::-1]

    # print(dubs)

    # from the doubles, generate known piece of the original uint64
    generated = []
    for idx in range(len(dubs)):
        if browser == 'chrome':
            recovered = struct.unpack('<Q', struct.pack('d', dubs[idx] + 1))[0] & (MASK >> 12)
        elif browser == 'firefox':
            recovered = dubs[idx] * (0x1 << 53)
        elif browser == 'safari':
            recovered = dubs[idx] / (1.0 / (1 << 53))
        generated.append(recovered)

    # setup symbolic state for xorshift128+
    ostate0, ostate1 = BitVecs('ostate0 ostate1', 64)
    sym_state0 = ostate0
    sym_state1 = ostate1
    slvr = Solver()
    conditions = []

    # run symbolic xorshift128+ algorithm for three iterations
    # using the recovered numbers as constraints
    for ea in range(len(dubs)):
        sym_state0, sym_state1, ret_conditions = sym_xs128p(slvr, sym_state0, sym_state1, generated[ea], browser)
        conditions += ret_conditions

    if slvr.check(conditions) == sat:
        # get a solved state
        m = slvr.model()
        state0 = m[ostate0].as_long()
        state1 = m[ostate1].as_long()
        slvr.add(Or(ostate0 != m[ostate0], ostate1 != m[ostate1]))
        if slvr.check(conditions) == sat:
            print('WARNING: multiple solutions found! use more dubs!')
        # print('state', state0, state1)

        generated = []
        # generate random numbers from recovered state
        for idx in range(generate_len):
            if browser == 'chrome':
                state0, state1, out = xs128p_backward(state0, state1, browser)
                out = state0 & MASK
            else:
                state0, state1, out = xs128p(state0, state1, browser)

            double = to_double(browser, out)
            # print('gen', double)
            generated.append(double)

        # use generated numbers to predict powerball numbers
        # power_ball(generated, browser)
        return generated
    else:
        print('UNSAT')
        return


def main():
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('--type', type=str, choices=['lehmer64', 'mt19937', 'lcg'], default='lehmer64')
    parser.add_argument('--input', type=str, default='rng_input.txt')
    parser.add_argument('--output', type=str, default='rng_output.txt')
    parser.add_argument('--predict_len', type=int, default=20)
    random_numbers = []
    args = parser.parse_args()
    with open(args.input, 'r') as f:
        for line in f:
            for num in line.strip().split(", "):
                random_numbers.append(float(num))
            # random_numbers.append(int(line.strip()))
    print(random_numbers)
    predict = crack(random_numbers, args.predict_len)
    if predict is not None:
        print(predict)
        with open(args.output, 'w') as f:
            for num in predict[:-1]:
                f.write(str(num) + ", ")
            f.write(str(predict[-1]))
            f.write("\n")

# Note:
# Open the browser, press F12 to open the concole, and input the following code:
# _ = []; for(var i=0; i<20; ++i) { _.push(Math.random()) } ; console.log(_)
# Copy and paste first n numbers (recommend >5) and compare the predicted outputs with true outputs


if __name__ == '__main__':
    main()
