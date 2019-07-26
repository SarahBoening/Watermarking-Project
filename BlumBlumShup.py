# Implements all needed functionality for the Blum Blum Shup random number generator
import numpy as np


def bbs(n, xi, m):
    seed = xi
    bits = 0
    for i in range(1, n):
        seed = pow(seed, 2) % m
        bits += pow(2, n-i) * (seed % 2)
    return bits, seed


def isBBSSeed(p, q, s):
    isIt = True
    if s % p == 0:
        isIt = False
    elif s&q == 0:
        isIt = True
    return isIt


def getBBSPath(num, xi, m, max_x, max_y):
    s = (2, num)
    path = np.zeros(s, dtype=int)
    for i in range(0, num-1):
        # generate coordinates (x,y)
        offset_x, xi = bbs(64, xi, m)
        offset_x = offset_x % max_x
        offset_y, xi = bbs(64, xi, m)
        offset_y = offset_y % max_y
        # check if coordinates were already used
        isNotOk = True
        while isNotOk:
            isNotOk = False
            if i != 0:
                for j in range(0, i-1):
                    if path[0, i] == offset_x and path[1, i] == offset_y:
                        isNotOk = True
            # was already used -> generate new coordinates
            if isNotOk:
                offset_x, xi = bbs(64, xi, m)
                offset_x = offset_x % max_x
                offset_y, xi = bbs(64, xi, m)
                offset_y = offset_y % max_y
        # save coordinates
        path[0, i] = offset_x
        path[1, i] = offset_y

    return path
