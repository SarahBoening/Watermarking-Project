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
        hashval +=('0' * (l - len(hashval)+2))
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
                # Cb color channel is at dim 2
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
        m = p*q
        xi = 20151208
        temp = np.array(c)
        path = bbs.getBBSPath(l, xi, m, temp.shape[0],temp.shape[1])
        i = 0
        stop = False # stop when finish embedding all bits in wm
        for j in range(0, path.shape[0]):
            for k in range(0, path.shape[1]):
                # get the next block position to embed
                m = path[0, i]
                n = path[1, i]
                # Cb color channel is at dim 2
                if b[i] == 1:
                    d[m, n, 2] = d[m, n, 2] * (1 + alpha * wm[0, i])
                elif b[i] == 0:
                    d[m, n, 2] = d[m, n, 2] * (1 - alpha * wm[0, i])
                i += 1
                if i >= l:
                    stop = True
                    break
            if stop:
                break

    # step 4: compute the watermarked image
    s = dct.jpgInverseDCT(d, x, y)
    return s
