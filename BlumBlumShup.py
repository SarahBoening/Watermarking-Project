# Implements all needed functionality for the Blum Blum Shup random number generator
import numpy as np

def bbs(n, xi, m):
    """
    Generate the next coefficient.

    n: number of bits for the coefficient
    xi: current seed
    m: modulo
    returns: next random coefficient, next seed
    """
    seed = xi
    bits = 0
    for i in range(1, n):
        seed = pow(seed, 2) % m
        bits += pow(2, n-i) * (seed % 2)
    return bits, seed

def getBBSPath(num, xi, m, max_x, max_y):
    """
    Generate the path of indices [row, column] to use for embedding.

    num: length of the path
    xi: initial seed
    m: modulo
    max_x: maximum value of x
    max_y: maximum value of y
    returns: path e.g [[r1,c1],[r2,c2],...]
    """
    s = (2, num)
    # initialize path
    path = np.zeros(s, dtype=int)
    for i in range(0, num):
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
                for j in range(0, i):
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
