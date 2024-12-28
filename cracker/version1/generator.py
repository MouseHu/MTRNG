import argparse


class LCG(object):
    def __init__(self, seed, m=672257317069504227, c=7382843889490547368, n=9223372036854775783, ):
        self.m = m
        self.c = c
        self.n = n
        self.state = seed  # the "seed"

    def next(self):
        self.state = (self.state * self.m + self.c) % self.n
        return self.state


class MT19937:
    w, n = 32, 624
    f = 1812433253
    m, r = 397, 31
    a = 0x9908B0DF
    d, b, c = 0xFFFFFFFF, 0x9D2C5680, 0xEFC60000
    u, s, t, l = 11, 7, 15, 18

    def __init__(self, seed):
        self.X = [0] * self.n
        self.cnt = 0
        self.initialize(seed)

    def initialize(self, seed):
        self.X[0] = seed
        for i in range(1, self.n):
            self.X[i] = (self.f * (self.X[i - 1] ^ (self.X[i - 1] >> (self.w - 2))) + i) & ((1 << self.w) - 1)
        self.twist()

    def twist(self):
        for i in range(self.n):
            lower_mask = (1 << self.r) - 1
            upper_mask = (~lower_mask) & ((1 << self.w) - 1)
            tmp = (self.X[i] & upper_mask) + (self.X[(i + 1) % self.n] & lower_mask)
            tmpA = tmp >> 1
            if (tmp % 2):
                tmpA = tmpA ^ self.a
            self.X[i] = self.X[(i + self.m) % self.n] ^ tmpA
        self.cnt = 0

    def temper(self):
        if self.cnt == self.n:
            self.twist()
        y = self.X[self.cnt]
        y = y ^ ((y >> self.u) & self.d)
        y = y ^ ((y << self.s) & self.b)
        y = y ^ ((y << self.t) & self.c)
        y = y ^ (y >> self.l)
        self.cnt += 1
        return y & ((1 << self.w) - 1)


def lehmer64(seed=574389759345345, datasize=20):
    g_lehmer64_state = 574389759345345 & 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF  # 128-bit state

    def _random():
        nonlocal g_lehmer64_state
        g_lehmer64_state *= 0xda942042e4dd58b5
        g_lehmer64_state &= 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
        # print(g_lehmer64_state, g_lehmer64_state >> 64)
        return g_lehmer64_state, g_lehmer64_state >> 64

    full_data = [_random() for _ in range(datasize)]
    data = [x[1] for x in full_data]
    return data


def mt19937(seed=12345, datasize=20):
    mt = MT19937(seed)
    return [mt.temper() for _ in range(datasize)]


def lcg(seed=1234, m=672257317069504227, c=7382843889490547368, n=9223372036854775783, datasize=20):
    lcg = LCG(seed, m=m, c=c, n=n)
    return [lcg.next() for _ in range(datasize)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, choices=['lehmer64', 'mt19937', 'lcg'], default='lcg')
    parser.add_argument('--output', type=str, default='rng_input.txt')
    parser.add_argument('--output_test', type=str, default='rng_test.txt')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--m', type=int, default=672257317069504227)
    parser.add_argument('--c', type=int, default=7382843889490547368)
    parser.add_argument('--n', type=int, default=9223372036854775783)
    parser.add_argument('--generate_len', type=int, default=20)
    parser.add_argument('--split_len', type=int, default=10)

    args = parser.parse_args()
    if args.type == 'lehmer64':
        random_numbers = lehmer64(seed=args.seed, datasize=args.generate_len)
    elif args.type == 'mt19937':
        random_numbers = mt19937(seed=args.seed, datasize=args.generate_len)
    elif args.type == 'lcg':
        random_numbers = lcg(seed=args.seed, m=args.m, c=args.c, n=args.n, datasize=args.generate_len)
    else:
        return
    print(random_numbers)

    with open(args.output, 'w') as f:
        for num in random_numbers[:args.split_len]:
            f.write(str(num) + '\t')
        f.write("\n")

    with open(args.output_test, 'w') as f:
        for num in random_numbers[args.split_len:]:
            f.write(str(num) + '\t')
        f.write("\n")


if __name__ == '__main__':
    main()
