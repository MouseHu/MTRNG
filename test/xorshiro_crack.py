import z3
import binascii


# Code derived from https://gist.github.com/karanlyons/805dbcc9e898dbd17e06f2627d5f9111

def bin2chr(data):
    result = ''

    while data:
        char = data & 0xff
        result += chr(char)
        data >>= 8

    return result


class Xoroshiro128Plus(object):
    def __init__(self, seed):
        self.seed = seed

    @staticmethod
    def rotl(x, k):
        return ((x << k) & 0xffffffffffffffff) | (x >> 64 - k)

    def next(self):
        s0 = self.seed[0]
        s1 = self.seed[1]

        result = (s0 + s1) & 0xffffffffffffffff

        s1 ^= s0

        self.seed[0] = self.rotl(s0, 55) ^ s1 ^ ((s1 << 14) & 0xffffffffffffffff)
        self.seed[1] = self.rotl(s1, 36)

        return result


def sym_xoroshiro128plus(solver, sym_s0, sym_s1, mask, result):
    s0 = sym_s0
    s1 = sym_s1
    sym_r = (sym_s0 + sym_s1)

    condition = z3.Bool('c0x%0.16x' % result)
    solver.add(z3.Implies(condition, (sym_r & mask) == result & mask))

    s1 ^= s0
    sym_s0 = z3.RotateLeft(s0, 55) ^ s1 ^ (s1 << 14)
    sym_s1 = z3.RotateLeft(s1, 36)

    return sym_s0, sym_s1, condition


def find_seed(res_masks):
    start_s0, start_s1 = z3.BitVecs('start_s0 start_s1', 64)
    sym_s0 = start_s0
    sym_s1 = start_s1
    solver = z3.Solver()
    conditions = []

    for result, mask in res_masks:
        sym_s0, sym_s1, condition = sym_xoroshiro128plus(solver, sym_s0, sym_s1, mask, result)
        conditions.append(condition)

    if solver.check(conditions) == z3.sat:
        model = solver.model()
        return (model[start_s0].as_long(), model[start_s1].as_long())
    else:
        return None


word1 = b"An Apple"[::-1]
word2 = b"A Day"[::-1]
val1 = binascii.hexlify(word1)
val2 = binascii.hexlify(word2)
print(val1)
print(val2)
val1 = int.from_bytes(word1, byteorder='big')
val2 = int.from_bytes(word2, byteorder='big')

print("Solving ...")
res = find_seed([(0x0, 0x0), (val1, 0xffffffffffffffff), (val2, 0xfffffffffffffffff)])
print("Seeds", res)

generator = Xoroshiro128Plus([res[0], res[1]])

for i in range(10):
    val = generator.next()
    print(i, hex(val), '\t', repr(bin2chr(val)))