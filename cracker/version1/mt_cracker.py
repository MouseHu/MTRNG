def get_bit(x, i):
    return (x & (1 << (MT19937.w - i - 1)))


def reverse_bits(x):
    rev = 0
    for i in range(MT19937.w):
        rev = (rev << 1)
        if (x > 0):
            if (x & 1 == 1):
                rev = (rev ^ 1)
            x = (x >> 1)
    return rev


def inv_left(y, a, b):
    return reverse_bits(inv_right(reverse_bits(y), a, reverse_bits(b)))


def inv_right(y, a, b):
    x = 0
    for i in range(MT19937.w):
        if (i < a):
            x |= get_bit(y, i)
        else:
            x |= (get_bit(y, i) ^ ((get_bit(x, i - a) >> a) & get_bit(b, i)))
    return x


class MT19937:
    w, n = 32, 624
    f = 1812433253
    m, r = 397, 31
    a = 0x9908B0DF
    d, b, c = 0xFFFFFFFF, 0x9D2C5680, 0xEFC60000
    u, s, t, l = 11, 7, 15, 18

    def __init__(self, seed):
        self.X = [0] * MT19937.n
        self.cnt = 0
        self.initialize(seed)

    def initialize(self, seed):
        self.X[0] = seed
        for i in range(1, MT19937.n):
            self.X[i] = (MT19937.f * (self.X[i - 1] ^ (self.X[i - 1] >> (MT19937.w - 2))) + i) & ((1 << MT19937.w) - 1)
        self.twist()

    def twist(self):
        for i in range(MT19937.n):
            lower_mask = (1 << MT19937.r) - 1
            upper_mask = (~lower_mask) & ((1 << MT19937.w) - 1)
            tmp = (self.X[i] & upper_mask) + (self.X[(i + 1) % MT19937.n] & lower_mask)
            tmpA = tmp >> 1
            if (tmp % 2):
                tmpA = tmpA ^ MT19937.a
            self.X[i] = self.X[(i + MT19937.m) % MT19937.n] ^ tmpA
        self.cnt = 0

    def temper(self):
        if self.cnt == MT19937.n:
            self.twist()
        y = self.X[self.cnt]
        y = y ^ ((y >> MT19937.u) & MT19937.d)
        y = y ^ ((y << MT19937.s) & MT19937.b)
        y = y ^ ((y << MT19937.t) & MT19937.c)
        y = y ^ (y >> MT19937.l)
        self.cnt += 1
        return y & ((1 << MT19937.w) - 1)

    def untemper(self, y):
        x = y
        x = inv_right(x, MT19937.l, ((1 << MT19937.w) - 1))
        x = inv_left(x, MT19937.t, MT19937.c)
        x = inv_left(x, MT19937.s, MT19937.b)
        x = inv_right(x, MT19937.u, MT19937.d)
        return x


def mt19937(X, predict_len):
    def construct_rng(X):
        rng = MT19937(0)
        for i in range(MT19937.n):
            rng.X[i] = rng.untemper(X[i])
        rng.twist()
        return rng

    rng = construct_rng(X)
    i = MT19937.n
    all_correct = True
    while i < len(X):
        next_num = rng.temper()
        print(next_num, X[i], next_num == X[i])
        all_correct = all_correct and next_num == X[i]
        i += 1

    if all_correct:
        new_rng = construct_rng(X[len(X)-624:])
        return [new_rng.temper() for _ in range(predict_len)]
    else:
        return None


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
    predict = mt19937(random_numbers, args.predict_len)
    if predict is not None:
        print(predict)
        with open(args.output, 'w') as f:
            for num in predict:
                f.write(str(num) + "\t")
            f.write("\n")


if __name__ == '__main__':
    main()
