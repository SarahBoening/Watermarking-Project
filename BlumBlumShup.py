# Implements all needed functionality for the Blum BLum Shup generator
import numpy as np


def bbs(n, xi, m):
    seed = xi
    bits = 0
    for i in range(1, n):
        seed = (seed**2) % m
        bits = bits + 2**(n-i)* (seed % 2)

    return bits, seed


def isBBSSeed(p,q,s):
    isIt = True
    if s % p == 0:
        isIt = False
    elif s&q == 0:
        isIt = True
    return isIt


def getBBSPath(num, xi, m, max_x, max_y):
    path = np.zeros(2, num)
    for i in range(1,num):
        # generate coordinates (x,y)
        offset_x, xi = bbs(64, xi, m)
        offset_x = offset_x % max_x + 1
        offset_y, xi = bbs(64, xi, m)
        offset_y = offset_y % max_y + 1
        # check if coordinates were already used
        isNotOk = True
        while isNotOk:
            isNotOk = False
            if i != 1:
                for j in range(1,i):
                    if path[1,i] == offset_x and path[2,1] == offset_y:
                        isNotOk = True
            # was already used -> generate new coordinates
            if isNotOk:
                offset_x, xi = bbs(64, xi, m)
                offset_x = offset_x % max_x + 1
                offset_y, xi = bbs(64, xi, m)
                offset_y = offset_y % max_y + 1

        # save coordinates
        path[1,i] = offset_x
        path[2,i] = offset_y

    return path
