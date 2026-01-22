import numpy as np

# Truth table
# x1 x2 | s1 s2 | y
#  0  0 |  1  0 | 0
#  0  1 |  1  1 | 1
#  1  0 |  1  1 | 1
#  1  1 |  0  1 | 0

def AND(x, y):
    x = np.array([x, y])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def NAND(x, y):
    x = np.array([x, y])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1
    
def OR(x, y):
    x = np.array([x, y])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def XOR(x, y):
    s1 = NAND(x, y)
    s2 = OR(x, y)
    return AND(s1, s2)

def __main__():
    print("XOR")
    print(XOR(0, 0))
    print(XOR(1, 0))
    print(XOR(0, 1))
    print(XOR(1, 1))

if __name__ == "__main__":
    __main__()