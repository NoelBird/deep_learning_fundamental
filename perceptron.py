import numpy as np

def AND(x1, x2):
    x = np.array([x1, x2, 1])
    a = np.array([0.5, 0.5, -0.75]) # 마지막 항목은 bias입니다.
    tmp = np.sum(a*x)
    return tmp >= 0

def OR(x1, x2):
    x = np.array([x1, x2, 1])
    w = np.array([0.5, 0.5, -0.25]) # 마지막 항목은 bias입니다.
    tmp = np.sum(x*w)
    return tmp >= 0

def NAND(x1, x2):
    return not AND(x1, x2)

def NOR(x1, x2):
    return not OR(x1, x2)

def XOR(x1, x2):
    y1 = NAND(x1, x2)
    y2 = OR(x1, x2)
    return AND(y1, y2)

print(XOR(0, 0))
print(XOR(0, 1))
print(XOR(1, 0))
print(XOR(1, 1))