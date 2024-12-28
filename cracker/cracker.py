def lehmer64(X):
    r = round(2.64929081169728e-7 * X[0] + 3.51729342107376e-7 * X[1] + 3.89110109147656e-8 * X[2]) % (2 ** 128)
    s = round(3.12752538137199e-7 * X[0] - 1.00664345453760e-7 * X[1] - 2.16685184476959e-7 * X[2]) % (2 ** 128)
    t = round(3.54263598631140e-8 * X[0] - 2.05535734808162e-7 * X[1] + 2.73269247090513e-7 * X[2]) % (2 ** 128)
    u = (r * 1556524 + s * 2249380 + t * 1561981) % (2 ** 128)
    v = (r * 8429177212358078682 + s * 4111469003616164778 + t * 3562247178301810180) % (2 ** 128)
    state = (0xda942042e4dd58b5 * u + v) % (2 ** 128)
    state = (state * 0xdb76c43996e558d0bdfbbe1277f2430d) % (2 ** 128)
    return state >> 64


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


def mt19937(X):
    def construct_rng(X):
        rng = MT19937(0)
        for i in range(MT19937.n):
            rng.X[i] = rng.untemper(X[i])
        rng.twist()
        return rng

    rng = construct_rng(X)
    if rng.temper() == X[MT19937.n]:
        return rng
    else:
        return None
