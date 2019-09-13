import numpy as np

'''
File to add all methods that we need for the analysis
'''


def chisquaretestcorr(values):
    '''
    test for correlated pairs
    :param values: histogram counts
    :return: chi^2 value y
    '''
    y = 0
    num = int(np.floor(values.shape[0]/2))
    for i in range(1, num):
        ex = (values[2*i] + values[2*i+1]) / 2
        di = pow((values[2*i] - ex), 2)
        if ex != 0:
            y += di/ex
    print(i)
    return y


def chisquaretestuncorr(values):
    '''
    test for uncorrelated pairs
    '''
    y = 0
    num = int(np.floor(values.shape[0]/2))
    for i in range(1, num):
        ex = (values[2*i] + values[2*i-1]) / 2
        di = pow((values[2*i] - ex), 2)
        if ex != 0:
            y += di/ex
    return y


def subhistcount(values, min=0, max=255):
    return np.histogram(values, bins=list(range(min, max+2)))[0]
