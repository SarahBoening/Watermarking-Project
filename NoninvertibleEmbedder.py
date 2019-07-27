import numpy as np
import BlumBlumShup as bbs
import YCrCbDCT as dct
import imagehash


def hashimage(image, l):
    """
    Create binary hash for the image.

    image: image to create hash
    l: length of the hash string
    returns: array of bits
    """
    # hash an image to a binary array of len
    hashval = bin(int(str(imagehash.average_hash(image)), 16))
    # if hash value smaller than l add zeros at the end
    if l > len(hashval):
        hashval += ('0' * (l - len(hashval) + 2))
    # print(hashval); e.g 0b1100000011110001111111101111011011000001000000000000000000000000000000000000000000000000000000000000
    # convert the hash value to array of bits and return; e.g [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    return [int(d) for d in hashval[2:(l + 2)]]


def noninvertibleEmbedder(wm, c, embed_type='Normal', alpha=0.025):
    """
    Embedd the image using noninvertible algorithms from lecture 13.

    wm: watermark
    c: coverwork (the original image)
    embed_type: switch between embedding methods
        1. DCT - use DCT coefficients from 8x8 jpeg transformation of Cb color channel
        2. BBS - Blum, Blum, Shup to define the paths of the coefficients
        3. Standard case: most l significant coefficients
    returns: watermarked image
    """
    l = wm.shape[1]
    # step 1: hash l (size of watermark) bits based on cover work
    b = hashimage(c, l)
    # step 2: perform DCT to get coefficients in blocks
    d, x, y = dct.jpgDCT(c)
    # step 3: insert watermark using 2 variations of embedding type (BBS / DCT)
    if embed_type == 'DCT':
        '''
        Use the DCT-coefficients (upper left corner of the 8 x 8-block decomposition) of the Cb
        color channel after the standard jpg-transformation of all 8  8-block (except of the
        outer blocks) for the embedding.
        '''
        i = 0
        stop = False # stop when finish embedding all bits in wm
        # take the left upper corner coefficient of each block (except the outer ones)
        for j in range(8, d.shape[0] - 8, 8):
            for k in range(8, d.shape[1] - 8, 8):
                # embedd at Cb color channel (dimension 2)
                if b[i] == 1:
                    d[j, k, 2] = d[j, k, 2] * (1 + alpha * wm[0, i])
                elif b[i] == 0:
                    d[j, k, 2] = d[j, k, 2] * (1 - alpha * wm[0, i])
                i += 1
                if i >= l:
                    stop = True
                    break
            if stop:
                break
    elif embed_type == 'BBS':
        '''
        Use the PRNG of Blum, Blum, and Shup with the primes p = 59999 and q = 60107 to
        determine the ordering of the ` most significant coefficients for the embedding (and the
        detection). Use 20151208 as initial value of xi.
        '''
        p = 5999
        q = 60107
        m = p * q
        xi = 20151208
        temp = np.array(c)
        path = bbs.getBBSPath(l, xi, m, temp.shape[0], temp.shape[1])
        for p in range(0, path.shape[1]):
            # get the next block position to embed
            m = path[0][p]
            n = path[1][p]
            # currently also embedd at Cb color channel (dimension 2)
            if b[p] == 1:
                d[m, n, 2] = d[m, n, 2] * (1 + alpha * wm[0, p])
            else:
                d[m, n, 2] = d[m, n, 2] * (1 - alpha * wm[0, p])
    else:
        '''
        Standard case, take the l most significant coefficients
        '''
        # get indices of the most significant coefficients in d
        # sort and get flat list of indices
        temp = np.copy(d)
        i = (-d).argsort(axis=None, kind='mergesort')
        # convert flat list to DCT coefficients e.g(array([...]), array([...]), array([...])) - rows,columns,channel
        j = np.unravel_index(i, d.shape)
        for i in range(0, l):
            # embedding at l most significant coefficients
            if b[i] == 1:
                temp[j[0][i], j[1][i], j[2][i]] = temp[j[0][i], j[1][i], j[2][i]] * (1 + alpha * wm[0, i])
            elif b[i] == 0:
                temp[j[0][i], j[1][i], j[2][i]] = temp[j[0][i], j[1][i], j[2][i]] * (1 - alpha * wm[0, i])

    # step 4: compute the watermarked image by inverse DCT
    s = dct.jpgInverseDCT(temp, x, y)
    return s


def invertEmbedding(S, wm, b, l, x, y, embed_type='Normal', alpha=0.025):
    """
    Embedd the image using noninvertible algorithms from lecture 13.

    S: DCT of stegowork (the watermarked image)
    wm: fake watermark
    b: fake hash bitstring
    l: length of embed string
    x: number of rows
    y: number of columns
    embed_type: switch between embedding methods
        1. DCT - use DCT coefficients from 8x8 jpeg transformation of Cb color channel
        2. BBS - Blum, Blum, Shup to define the paths of the coefficients
        3. Standard case: most l significant coefficients
    returns: an image as invertion of embedding function
    """

    # calculate DCT of fake cover work
    '''
    Standard case, take the l most significant coefficients
    '''
    d = S
    C = np.copy(d)
    # get indices of the most significant coefficients in d
    # sort and get flat list of indices
    i = (-d).argsort(axis=None, kind='mergesort')
    # convert flat list to DCT coefficients e.g(array([...]), array([...]), array([...])) - rows,columns,channel
    j = np.unravel_index(i, d.shape)
    for i in range(0, l):
        # invert embedding at l most significant coefficients
        if b[i] == 1:
            C[j[0][i], j[1][i], j[2][i]] = C[j[0][i], j[1][i], j[2][i]] / (1 + alpha * wm[0, i])
        elif b[i] == 0:
            C[j[0][i], j[1][i], j[2][i]] = C[j[0][i], j[1][i], j[2][i]] / (1 - alpha * wm[0, i])

    # compute the fake coverwork by inverse DCT
    c = dct.jpgInverseDCT(C, x, y)
    return c
