from dataset.dataset import binary, LehmerForwardDataset

forward_dataset = LehmerForwardDataset(data_dir='../data/lehmer64_12.dat',
                                       state_data_dir='../data/lehmer64_state_12.pkl', split=[0.0, 1.0])

a = [1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1,
     0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1]
for i in range(64):
    a.insert(0, 0)


def multiplier(x, y):
    print(x, y)
    print(len(x))
    carry = 0
    output = []
    sum = 0
    for i in range(128):
        sum += x[i] * a[i]
        carry += sum / 2
        output.append(sum % 2)
    print(y, output)


for i in range(2):
    data = forward_dataset[i]
    x, y = data
    x, y = x.numpy(), y.numpy()
    multiplier(x, y)
    # print(data)
