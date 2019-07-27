import numpy as np
import BlumBlumShup as bbs
import YCrCbDCT as dct
import imagehash


def hashimage(image, l):
    # hash an image to a binary array of len
    hashval = bin(int(str(imagehash.average_hash(image)), 16))
    # if hash value smaller than l add zeros at the end
    if l > len(hashval):
        hashval +=('0' * (l - len(hashval)+2))
    return [int(d) for d in hashval[2:(l + 2)]]


def noninvertibleEmbedder(wm, c, embed_type='Normal', alpha=0.2):
    # implements the noninvertible Embedder from lecture 13
    # wm = watermark drawn independently from N(0, 1)-distribution., c = cover work
    # embed_type: switches between embedding methods

    l = wm.shape[1]
    # 1. hash l (size of watermark) bits based on cover work
    b = hashimage(c, l)
    # 2. perform DCT to get coefficients in blocks
    # d, h, w = jpgc.jpgDCT(c)
    d, x, y = dct.jpgDCT(c)

    # 3. VARIATIONS: 1. l most significant bit, 2.upper left corner not the outer ones 3. BBS path
    # 4. Embed based on variations
    if embed_type == 'BBS':
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
        stop = False
        for j in range(0, path.shape[0]):
            for k in range(0, path.shape[1]):
                if b[i] == 1:
                    d[j, k, 2] = d[j, k, 2] * (1 + alpha * wm[0, i])
                else:
                    d[j, k, 2] = d[j, k, 2] * (1 - alpha * wm[0, i])
                i += 1
                if i >= l:
                    stop = True
                    break
            if stop:
                break

    elif embed_type == 'DCT':
        '''
        Use the DCT-coefficients (upper left corner of the 8 x 8-block decomposition) of the Cb
        color channel after the standard jpg-transformation of all 8  8-block (except of the
        outer blocks) for the embedding.
        '''
        i = 0
        stop = False
        for j in range(8, d.shape[0] - 8, 8):
            for k in range(8, d.shape[1] - 8, 8):
                if b[i] == 1:
                    d[j, k, 2] = d[j, k, 2] * (1 + alpha * wm[0, i])
                else:
                    d[j, k, 2] = d[j, k, 2] * (1 - alpha * wm[0, i])
                i += 1
                if i >= l:
                    stop = True
                    break
            if stop:
                break
    else:
        '''
        standard case, take the l most significant coefficients
        '''
        # get indices of the most significant coefficients in d
        i = (-d).argsort(axis=None, kind='mergesort')
        j = np.unravel_index(i, d.shape)
        np.vstack(j)
        for i in range(0, l):
            if b[i] == 1:
                d[j[0][i], j[1][i], j[2][i]] = d[j[0][i], j[1][i], j[2][i]] * (1 + alpha * wm[0, i])
            else:
                d[j[0][i], j[1][i], j[2][i]] = d[j[0][i], j[1][i], j[2][i]] * (1 - alpha * wm[0, i])

    # 5. inverse DCT
    # s = jpgc.jpgInverseDCT(d, h, w)
    s = dct.jpgInverseDCT(d, x, y)

    # 6. return watermarked image
    return s
