class A(object):
    def __init__(self):
        self.do()

    def do(self):
        print("A")


class B(A):
    def __init__(self):
        super().__init__()

    def do(self):
        print("B")


x = B()

total = 0
for i in range(63):
    y = 63 - i
    total += y*(1/2)**(i+1)

print(total)
