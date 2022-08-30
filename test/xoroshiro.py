mystr = " Daniel Lemire  "

def convertstringtovalues(mystring):
    assert len(mystring) == 16
    array = [ord(c) for c in mystring]
    x, y = 0, 0
    for i in range(8):
        x = x + (array[i] << (64 - 8 * (7 - i)))
        y = y + (array[i + 8] << (64 - 8 * (7 - i)))
    return x,y


out1, out2 = convertstringtovalues(mystr)

print(out1, out2)