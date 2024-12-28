def get_state(X):
    r = round(2.64929081169728e-7 * X[0] + 3.51729342107376e-7 * X[1] + 3.89110109147656e-8 * X[2]) % (2 ** 128)
    s = round(3.12752538137199e-7 * X[0] - 1.00664345453760e-7 * X[1] - 2.16685184476959e-7 * X[2]) % (2 ** 128)
    t = round(3.54263598631140e-8 * X[0] - 2.05535734808162e-7 * X[1] + 2.73269247090513e-7 * X[2]) % (2 ** 128)
    u = (r * 1556524 + s * 2249380 + t * 1561981) % (2 ** 128)
    v = (r * 8429177212358078682 + s * 4111469003616164778 + t * 3562247178301810180) % (2 ** 128)
    state = (0xda942042e4dd58b5 * u + v) % (2 ** 128)
    state = (state * 0xdb76c43996e558d0bdfbbe1277f2430d) % (2 ** 128)
    return state


def lehmer64(random_numbers, predict_len):
    state = get_state(random_numbers[-3:])
    print(state)
    predicts = [state >> 64]
    for i in range(predict_len - 1):
        state *= 0xda942042e4dd58b5
        state &= 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
        predicts.append(state >> 64)
    return predicts

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
    predict = lehmer64(random_numbers, args.predict_len)
    if predict is not None:
        print(predict)
        with open(args.output, 'w') as f:
            for num in predict:
                f.write(str(num) + "\t")
            f.write("\n")


if __name__ == '__main__':
    main()
