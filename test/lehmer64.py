def lehmer64():
    g_lehmer64_state = 543543554 & 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF  # 128-bit state

    def _random():
        nonlocal g_lehmer64_state
        g_lehmer64_state *= 0xda942042e4dd58b5
        g_lehmer64_state &= 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
        # print("state: ", g_lehmer64_state, " output: ", g_lehmer64_state >> 64)
        return g_lehmer64_state >> 64

    def _state():
        nonlocal g_lehmer64_state
        return g_lehmer64_state

    return _random, _state


def reconstruct(X):
    r = round(2.64929081169728e-7 * X[0] + 3.51729342107376e-7 * X[1] + 3.89110109147656e-8 * X[2]) % (2 ** 128)
    s = round(3.12752538137199e-7 * X[0] - 1.00664345453760e-7 * X[1] - 2.16685184476959e-7 * X[2]) % (2 ** 128)
    t = round(3.54263598631140e-8 * X[0] - 2.05535734808162e-7 * X[1] + 2.73269247090513e-7 * X[2]) % (2 ** 128)
    u = (r * 1556524 + s * 2249380 + t * 1561981) % (2 ** 128)
    v = (r * 8429177212358078682 + s * 4111469003616164778 + t * 3562247178301810180) % (2 ** 128)
    state = (0xda942042e4dd58b5 * u + v) % (2 ** 128)
    state = (state * 0xdb76c43996e558d0bdfbbe1277f2430d) % (2 ** 128)
    return state >> 64


def reconstruct2(X):
    r = round(2.64929081169728e-7 * X[0] + 3.51729342107376e-7 * X[1] + 3.89110109147656e-8 * X[2]) % (2 ** 128)
    s = round(3.12752538137199e-7 * X[0] - 1.00664345453760e-7 * X[1] - 2.16685184476959e-7 * X[2]) % (2 ** 128)
    t = round(3.54263598631140e-8 * X[0] - 2.05535734808162e-7 * X[1] + 2.73269247090513e-7 * X[2]) % (2 ** 128)
    # u = (r * 1556524 + s * 2249380 + t * 1561981) % (2 ** 128)
    # v = (r * 8429177212358078682 + s * 4111469003616164778 + t * 3562247178301810180) % (2 ** 128)
    # state = (0xda942042e4dd58b5 * u + v) % (2 ** 128)
    state = (0xa4ec846cd16980ea410f6f4ab8ef517e * r + 0x675b7735fe063f6c61ab66b688363706 * s + 0x4aa11090ca91cfdb234ff1639d0d3721 * t) % (
                        2 ** 128)
    return state >> 64


lhemer64_generator, state_printer = lehmer64()
print(state_printer)
start = 30
x = [lhemer64_generator() for _ in range(start + 4)]

print(x)
# print(state_printer())
the_state = reconstruct(x[start:start + 3])
the_state2 = reconstruct2(x[start:start + 3])

print(the_state)
print(the_state2)
print(x[start + 3])
