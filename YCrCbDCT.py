import scipy
from scipy import signal
import numpy as np
import cv2

# code is mostly based on https://inst.eecs.berkeley.edu/~ee123/sp16/Sections/JPEG_DCT_Demo.html
# and https://www.hdm-stuttgart.de/~maucher/Python/MMCodecs/html/jpegUpToQuant.html

# for YCrCb DCT
QY = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
               [12, 12, 14, 19, 26, 48, 60, 55],
               [14, 13, 16, 24, 40, 57, 69, 56],
               [14, 17, 22, 29, 51, 87, 80, 62],
               [18, 22, 37, 56, 68, 109, 103, 77],
               [24, 35, 55, 64, 81, 104, 113, 92],
               [49, 64, 78, 87, 103, 121, 120, 101],
               [72, 92, 95, 98, 112, 100, 103, 99]])

QC = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
               [18, 21, 26, 66, 99, 99, 99, 99],
               [24, 26, 56, 99, 99, 99, 99, 99],
               [47, 66, 99, 99, 99, 99, 99, 99],
               [99, 99, 99, 99, 99, 99, 99, 99],
               [99, 99, 99, 99, 99, 99, 99, 99],
               [99, 99, 99, 99, 99, 99, 99, 99],
               [99, 99, 99, 99, 99, 99, 99, 99]])

Q2 = np.zeros((8, 8, 3))
Q2[:, :, 0] = QY
Q2[:, :, 1] = QC
Q2[:, :, 2] = QC


def dct2(a):
    return scipy.fftpack.dct(scipy.fftpack.dct(a, axis=0, norm='ortho'), axis=1, norm='ortho')


def idct2(a):
    return scipy.fftpack.idct(scipy.fftpack.idct(a, axis=0, norm='ortho'), axis=1, norm='ortho')


def jpgDCT(image):
    """
    Implement standard JPEG encoding.

    image: image array in RGB format
    returns: DCT coefficients after compression, number of rows, number of columns
    """
    # get image array as numpy and get shape
    im = np.array(image)
    x = im.shape[0]
    y = im.shape[1]

    # convert image from RGB to YCrCb
    im = cv2.cvtColor(im.astype('float32'), cv2.COLOR_RGB2YCR_CB)
    # crop image to 8x8 divisible size
    h, w = np.round(np.array(im.shape[:2]) / 8) * 8
    im = im[:int(h), :int(w), :]
    # shift the pixels values of all channels to [-128,...,127]
    im -= 128
    # new shape
    imsize = im.shape
    # initialize DCT coefficients with same size as the image
    dct = np.zeros(imsize, dtype='float32')

    # 8x8 blocks for DCT
    for i in np.r_[:imsize[0]:8]:
        for j in np.r_[:imsize[1]:8]:
            # apply 2D-DCT to current block
            currentBlock = dct2(im[i:(i + 8), j:(j + 8)])
            # quantize DCT coefficients
            currentBlock = np.round(currentBlock / Q2)
            # replace DCT coefficients with new values
            dct[i:(i + 8), j:(j + 8)] = currentBlock
    return dct, x, y


def jpgInverseDCT(coeffs, x, y):
    """
    Implement standard JPEG decoding.

    coeffs: DCT coefficients of decoded image
    x: number of rows
    y: number of columns
    returns: image after inverse DCT transform
    """
    imsize = coeffs.shape
    im_dct = np.zeros(imsize, dtype='float32')

    # 8x8 blocks for IDCT
    for i in np.r_[:imsize[0]:8]:
        for j in np.r_[:imsize[1]:8]:
            # inverse quantization
            coeffs[i:(i + 8), j:(j + 8)] *= Q2
            # apply 2D-IDCT to current block
            currentBlock = idct2(coeffs[i:(i + 8), j:(j + 8)])
            # inverver shifting of pixel values to [0,...,255]
            im_dct[i:(i + 8), j:(j + 8)] = currentBlock + 128

    # convert image from RGB to YCrCb
    rgb = np.resize(im_dct, (int(x), int(y), 3))
    rgb = cv2.cvtColor(rgb.astype('float32'), cv2.COLOR_YCR_CB2RGB)
    # replace out-of-range values with 0 and 255
    max_indices = rgb > 255
    rgb[max_indices] = 255
    min_indices = rgb < 0
    rgb[min_indices] = 0
    return rgb
