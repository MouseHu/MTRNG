uint64_t s[2];

uint64_t set_seed(uint64_t a, uint64_t b) {
    s[0] = a;
    s[1] = b;
}

uint64_t next(void) {
    uint64_t s1 = s[0];
    const uint64_t s0 = s[1];
    s[0] = s0;
    s1 ^= s1 << 23; // a
    s[1] = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5); // b, c
    return s[1] + s0;
}