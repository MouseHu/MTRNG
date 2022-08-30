import numba
import numpy as np

log_data_size = 24
data_size = 2 ** log_data_size


# @numba.jit(nopython=True)
def lehmer64(datasize=data_size):
    g_lehmer64_state = 574389759345345 & 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF  # 128-bit state

    def _random():
        nonlocal g_lehmer64_state
        g_lehmer64_state *= 0xda942042e4dd58b5
        g_lehmer64_state &= 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
        # print(g_lehmer64_state, g_lehmer64_state >> 64)
        return g_lehmer64_state >> 64

    data = [_random() for _ in range(data_size)]
    return data


@numba.jit(nopython=True)
def xorshift128plus(datasize=data_size):
    '''xorshift+
    https://en.wikipedia.org/wiki/Xorshift#xorshift+
    シフト演算で使用している 3 つの数値は元論文 Table.1 参照
    http://vigna.di.unimi.it/ftp/papers/xorshiftplus.pdf
    doi:10.1016/j.cam.2016.11.006
    '''

    s0 = 1
    s1 = 2

    def _random():
        nonlocal s0, s1
        x, y = s0, s1
        x = x ^ ((x << 23) & 0xFFFFFFFFFFFFFFFF)  # 64bit
        x = (x ^ (x >> 18)) ^ (y ^ (y >> 5))
        s0, s1 = y, x
        return (s0 + s1) & 0xFFFFFFFFFFFFFFFF  # 64-bit

    data = [_random() for _ in range(data_size)]
    # return _random
    return data


@numba.jit(nopython=True)
def xorshift128(datasize=data_size):
    '''xorshift
    https://ja.wikipedia.org/wiki/Xorshift
    '''

    x = 123456789
    y = 362436069
    z = 521288629
    w = 88675123

    def _random():
        nonlocal x, y, z, w
        t = x ^ ((x << 11) & 0xFFFFFFFF)  # 32bit
        x, y, z = y, z, w
        w = (w ^ (w >> 19)) ^ (t ^ (t >> 8))
        return w

    data = [_random() for _ in range(data_size)]
    # return _random
    return data


def main():
    name = "lehmer64"

    if name == "xorshift128":
        data = xorshift128()
    elif name == "lehmer64":
        data = lehmer64()
    else:
        assert name == "xorshift128plus"
        data = xorshift128plus()
    print(max(data))
    data = np.array(data, dtype=np.uint64)
    print(data)
    data.tofile(f'../data/{name}_{log_data_size}.dat')


if __name__ == '__main__':
    main()
