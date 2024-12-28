import math
from functools import reduce


def crack_unknown_increment(states, modulus, multiplier):
    increment = (states[1] - states[0] * multiplier) % modulus
    return modulus, multiplier, increment


def crack_unknown_multiplier(states, modulus):
    print(states, modulus)
    delta10 = (states[1] - states[0]) % modulus
    multiplier = (states[2] - states[1]) * modinv(delta10, modulus) % modulus
    return crack_unknown_increment(states, modulus, multiplier)


def egcd(a, b):
    if a == 0:
        return (b, 0, 1)
    else:
        g, x, y = egcd(b % a, a)
        return (g, y - (b // a) * x, x)


def modinv(b, n):
    # calulate the modular inverse of b mod n
    g, x, _ = egcd(b, n)
    if g == 1:
        return x % n
    else:
        print(1111)


def lcg(states):
    diffs = [s1 - s0 for s0, s1 in zip(states, states[1:])]
    zeroes = [t2 * t0 - t1 * t1 for t0, t1, t2 in zip(diffs, diffs[1:], diffs[2:])]
    modulus = abs(reduce(math.gcd, zeroes))
    print(modulus)
    return crack_unknown_multiplier(states, modulus)


def predictor(states, generate_len=10):
    n, m, c = lcg(states)
    print(m, c, n)
    new_rng = LCG(states[-1], m, c, n)
    return [new_rng.next() for _ in range(generate_len)]


class LCG(object):
    m = 672257317069504227  # the "multiplier"
    c = 7382843889490547368  # the "increment"
    n = 9223372036854775783  # the "modulus"

    def __init__(self, seed, m=672257317069504227, c=7382843889490547368, n=9223372036854775783, ):
        self.m = m
        self.c = c
        self.n = n
        self.state = seed  # the "seed"

    def next(self):
        self.state = (self.state * self.m + self.c) % self.n
        return self.state


def main():
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('--type', type=str, choices=['lehmer64', 'mt19937', 'lcg'], default='lehmer64')
    parser.add_argument('--input', type=str, default='rng_input.txt')
    parser.add_argument('--output', type=str, default='rng_output.txt')
    parser.add_argument('--predict_len', type=int, default=10)
    random_numbers = []
    args = parser.parse_args()
    with open(args.input, 'r') as f:
        for line in f:
            for num in line.strip().split("\t"):
                random_numbers.append(int(num))
            # random_numbers.append(int(line.strip()))
    print(random_numbers)
    predict = predictor(random_numbers, args.predict_len)
    if predict is not None:
        print(predict)
        with open(args.output, 'w') as f:
            for num in predict:
                f.write(str(num) + "\t")
            f.write("\n")


if __name__ == '__main__':
    main()
