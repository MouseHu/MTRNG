import randomgen
# from numpy.random import Generator
import numpy as np

log_data_size = 24
data_size = 2 ** log_data_size
prng_name = "PCG64"
prng_dict ={
    "xoroshiro128plus": randomgen.Xoroshiro128,
    "PCG64":randomgen.pcg64.PCG64
}
if __name__ == "__main__":
    print(log_data_size)
    RNG = randomgen.xoroshiro128.Xoroshiro128
    rng = RNG(seed=0)
    rand_numbers = rng.random_raw(data_size, output=True)
    rand_numbers = np.array(rand_numbers)
    print(rand_numbers[:10])
    print(rand_numbers.dtype)
    print("Dumping Data")
    rand_numbers.tofile(f"../data/{prng_name}_{log_data_size}.dat")
    print("Done")
