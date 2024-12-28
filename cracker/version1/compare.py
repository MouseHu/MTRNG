import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--test_file", type=str, default="rng_test.txt")
parser.add_argument("--predict_file", type=str, default="rng_output.txt")
parser.add_argument("--type", type=str, default="xorshiro", choices=["lcg","lehmer64", "mt19937", "xorshiro"])

args = parser.parse_args()

splitter = ", " if args.type == "xorshiro" else "\t"
test= []
with open(args.test_file, "r") as f:
    for line in f:
        for num in line.strip().split(splitter):
            test.append(float(num))

predict = []
with open(args.predict_file, "r") as f:
    for line in f:
        for num in line.strip().split(splitter):
            predict.append(float(num))
print(len(test), len(predict))
for i in range(len(test)):
    if test[i] != predict[i]:
        print(i, test[i], predict[i])
        print("Not equal!")
        exit()
print("All equal!")
