# Random Number Generation

In general, we generate a sequence of random numbers and split them into two files, `rng_input.txt` and `rng_test.txt`.
We use `rng_input.txt` as the input to the cracker and `rng_test.txt` as the test file to verify the correctness of the
cracker.

To run the code, first install the required packages via

```
pip install -r requirements.txt
```

## Linear Congruential Generator

You can generate LCG by run the following code

```
python generator.py --type lcg --seed 12345 --m=672257317069504227 --c 7382843889490547368 --n 9223372036854775783,
```

You can change seed, m, c and n as you like.

## MT19937

MT19937 is the default rng in python and it is known to have a long period and pass strong statistical tests. You can
generate MT19937 by run the following code

```
python generator.py --type mt19937 --seed 12345 --generate_len 660 --split_len 650
```

You can change seed as you like. Note that we need more than 623 numbers to crack MT19937 so   `split_len` must be
greater than 623.

## Lehmer64

Lehmer64 is the fastest RNG that passes PractRand. You can generate Lehmer64 by run the following code

```
python generator.py --type lehmer64 --seed 12345
```

You can change seed as you like.

## Xorshiro

Xorshiro128+ is the default RNG in javascript. To generate Xorshiro128+, you can open Chrome, press F12 to open the
console and run the following code

```
_ = []; for(var i=0; i<20; ++i) { _.push(Math.random()) } ; console.log(_)
_ = []; for(var i=0; i<20; ++i) { _.push(Math.random()) } ; console.log(_)
```

This code snippet with generate two lists of 20 random numbers and print them to the console. You can copy and paste the
output to `rng_input.txt` and  `rng_test.txt` manually (do not include the brackets). You can change the seed if you like.

# Cracking RNGs

Run the respective cracker for each RNG. For example, to crack MT19937, run

```
python mt_cracker.py --input rng_input.txt --output rng_output.txt --predict_len 10
```

Then run the following code to verify the correctness of the cracker

```
python compare.py --predict_file rng_output.txt --test_file rng_test.txt --type mt19937
```

