from cracker.cracker_func import mt19937
import random


# print(random.getrandbits(32))

def generate_sequence(N):
    seq = []
    for i in range(N):
        seq.append(random.getrandbits(32))
    return seq


seq = generate_sequence(1000)
print(mt19937(seq))
