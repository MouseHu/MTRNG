import randomgen
# from numpy.random import Generator
import numpy as np

log_data_size = 24
data_size = 2 ** log_data_size
prng_name = "PCG32"
prng_dict = {
    "xoroshiro128plus": randomgen.Xoroshiro128,
    "xorshift1024": randomgen.Xorshift1024,
    "PCG64": randomgen.pcg64.PCG64,
    "PCG32": randomgen.pcg32.PCG32,
    "xoroshiro128plusplus": randomgen.Xoroshiro128
}
if __name__ == "__main__":
    print(log_data_size)
    RNG = prng_dict[prng_name]
    if prng_name == "xoroshiro128plusplus":
        rng = RNG(seed=0, plusplus=True)
    else:
        rng = RNG(seed=0)
    rand_numbers = rng.random_raw(data_size, output=True)
    rand_numbers = np.array(rand_numbers)
    print(rand_numbers[:10])
    print(rand_numbers.dtype)
    print("Dumping Data")
    rand_numbers.tofile(f"../data/{prng_name}_{log_data_size}.dat")
    print("Done")
