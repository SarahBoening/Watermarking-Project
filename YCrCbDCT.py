import scipy
from scipy import signal
import numpy as np
import cv2

# code is mostly based on https://inst.eecs.berkeley.edu/~ee123/sp16/Sections/JPEG_DCT_Demo.html
# and https://www.hdm-stuttgart.de/~maucher/Python/MMCodecs/html/jpegUpToQuant.html

# for RGB DCT
M = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
              [12, 12, 14, 19, 26, 48, 60, 55],
              [14, 13, 16, 24, 40, 57, 69, 56],
              [14, 17, 22, 29, 51, 87, 80, 62],
              [18, 22, 37, 56, 68, 109, 103, 77],
              [24, 35, 55, 64, 81, 104, 113, 92],
              [49, 64, 78, 87, 103, 121, 120, 101],
              [72, 92, 95, 98, 112, 100, 103, 99]])

Q = np.zeros((M.shape[0], M.shape[1], 3))
Q[:, :, 0] = M
Q[:, :, 1] = M
Q[:, :, 2] = M

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
    # im = np.array(image, dtype='int8')
    im = np.array(image)
    x = im.shape[0]
    y = im.shape[1]

    im = cv2.cvtColor(im.astype('float32'), cv2.COLOR_RGB2YCR_CB)

    h, w = np.round(np.array(im.shape[:2]) / 8) * 8
    im = im[:int(h), :int(w), :]
    im -= 128
    imsize = im.shape
    dct = np.zeros(imsize, dtype='float32')

    # 8x8 blocks for DCT
    for i in np.r_[:imsize[0]:8]:
        for j in np.r_[:imsize[1]:8]:
            currentBlock = dct2(im[i:(i + 8), j:(j + 8)])
            # currentBlock = np.round(currentBlock)
            currentBlock = np.round(currentBlock / Q2)
            dct[i:(i + 8), j:(j + 8)] = currentBlock
    return dct, x, y


def jpgInverseDCT(coeffs, x, y):
    imsize = coeffs.shape
    im_dct = np.zeros(imsize, dtype='float32')
    for i in np.r_[:imsize[0]:8]:
        for j in np.r_[:imsize[1]:8]:
            coeffs[i:(i + 8), j:(j + 8)] *= Q2
            currentBlock = idct2(coeffs[i:(i + 8), j:(j + 8)])
            im_dct[i:(i + 8), j:(j + 8)] = currentBlock + 128

    rgb = np.resize(im_dct, (int(x), int(y), 3))
    rgb = cv2.cvtColor(rgb.astype('float32'), cv2.COLOR_YCR_CB2RGB)

    max_indices = rgb > 255
    rgb[max_indices] = 255
    min_indices = rgb < 0
    rgb[min_indices] = 0

    return rgb


def rgbDCT(image):
    im = np.array(image, dtype='int8')
    x = im.shape[0]
    y = im.shape[1]
    h, w = np.round(np.array(im.shape[:2]) / 8) * 8
    im = im[:int(h), :int(w), :]
    im -= 128
    imsize = im.shape
    dct = np.zeros(imsize, dtype='float32')
    # 8x8 blocks for DCT
    for i in np.r_[:imsize[0]:8]:
        for j in np.r_[:imsize[1]:8]:
            currentBlock = dct2(im[i:(i + 8), j:(j + 8)])
            # currentBlock = np.round(currentBlock)
            currentBlock = np.round(currentBlock / Q)
            dct[i:(i + 8), j:(j + 8)] = currentBlock
    return dct, x, y


def rgbInverseDCT(coeffs, x, y):
    imsize = coeffs.shape
    im_dct = np.zeros(imsize, dtype='float32')
    for i in np.r_[:imsize[0]:8]:
        for j in np.r_[:imsize[1]:8]:
            coeffs[i:(i + 8), j:(j + 8)] *= Q
            currentBlock = idct2(coeffs[i:(i + 8), j:(j + 8)])
            im_dct[i:(i + 8), j:(j + 8)] = currentBlock + 128

    rgb = np.resize(im_dct, (int(x), int(y), 3))

    max_indices = rgb > 255
    rgb[max_indices] = 255
    min_indices = rgb < 0
    rgb[min_indices] = 0

    return rgb
