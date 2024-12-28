import math
from functools import reduce


def crack_unknown_increment(states, modulus, multiplier):
    increment = (states[1] - states[0] * multiplier) % modulus
    return modulus, multiplier, increment


def crack_unknown_multiplier(states, modulus):
    multiplier = (states[2] - states[1]) * modinv(states[1] - states[0], modulus) % modulus
    return crack_unknown_increment(states, modulus, multiplier)


def egcd(a, b):
    if a == 0:
        return (b, 0, 1)
    else:
        g, x, y = egcd(b % a, a)
        return (g, y - (b // a) * x, x)


def modinv(b, n):
    g, x, _ = egcd(b, n)
    if g == 1:
        return x % n


def lcg(states):
    diffs = [s1 - s0 for s0, s1 in zip(states, states[1:])]
    zeroes = [t2 * t0 - t1 * t1 for t0, t1, t2 in zip(diffs, diffs[1:], diffs[2:])]
    modulus = abs(reduce(math.gcd, zeroes))
    return crack_unknown_multiplier(states, modulus)


def predict(states, generate_len=10):
    n, m, c = lcg(states)
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


rng = LCG(12345)

X = [rng.next() for i in range(20)]

print(X)
print(X[10:])
print(lcg(X[:10]))
print(predict(X[:10]))
