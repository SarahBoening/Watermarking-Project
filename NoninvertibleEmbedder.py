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


def nonInvertibleEmbedder(wm, c, alpha=0.04):
    """
    Embed the image using noninvertible algorithms from lecture 13.

    wm: watermark
    c: coverwork (the original image)
    returns: watermarked image
    """
    l = wm.shape[1]
    # step 1: hash l (size of watermark) bits based on cover work
    b = hashimage(c, l)
    # step 2: perform DCT to get coefficients in blocks
    d, x, y = dct.jpgDCT(c)
    # step 3: insert watermark using 2 variations of embedding type (BBS / DCT)
    p = 5999
    q = 60107
    Mb = p * q
    xi = 20151208
    path = bbs.getDCTBBSPath(l, xi, Mb, d.shape[1] - 8, d.shape[0] - 8)
    np.save("path", path)
    for i in range(0, path.shape[1]):
        # get the next block position to embed
        n = path[0][i]
        m = path[1][i]
        # currently also embed at Cb color channel (dimension 2)
        if b[i] == 1:
            d[m, n, 2] = d[m, n, 2] * (1 + alpha * wm[0, i])
        elif b[i] == 0:
            d[m, n, 2] = d[m, n, 2] * (1 - alpha * wm[0, i])
    # step 4: compute the watermarked image by inverse DCT
    s = dct.jpgInverseDCT(d, x, y)
    return s


def invertEmbedding(S, wm, b, l, x, y, alpha=0.04, sameSeed=False):
    """
    Embed the image using noninvertible algorithms from lecture 13.

    S: DCT of stegowork (the watermarked image)
    wm: fake watermark
    b: fake hash bitstring
    l: length of embed string
    x: number of rows
    y: number of columns
    returns: an image as invertion of embedding function
    """
    # calculate DCT of fake cover work
    C = np.copy(S)
    # same seed for bbs as embedder
    if sameSeed:
        p = 5999
        q = 60107
        Mb = p * q
        xi = 20151208
    else:
        p = 9539
        q = 54193
        Mb = p * q
        xi = 201981536
    path = bbs.getDCTBBSPath(l, xi, Mb, C.shape[1] - 8, C.shape[0] - 8)
    for i in range(0, path.shape[1]):
        # get the next block position to embed
        n = path[0][i]
        m = path[1][i]
        # currently also invert embedding at Cb color channel (dimension 2)
        if b[i] == 1:
            C[m, n, 2] = C[m, n, 2] / (1 + alpha * wm[0, i])
        else:
            C[m, n, 2] = C[m, n, 2] / (1 - alpha * wm[0, i])
    # compute the fake coverwork by inverse DCT
    c = dct.jpgInverseDCT(C, x, y)
    return c
