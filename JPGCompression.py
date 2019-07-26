import numpy as np
import cv2
import matplotlib.pyplot as plt

'''
Source: https://www.hdm-stuttgart.de/~maucher/Python/MMCodecs/html/jpegUpToQuant.html
'''

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
# quality factor
scale = 1.0
Q = [QY * scale, QC * scale, QC * scale]
B = 8  # blocksize


def jpgDCT(image):
    img1 = np.array(image)
    h, w = np.array(img1.shape[:2]) / B * B
    img1 = img1[:int(h), :int(w)]
    # transform RGB to Ycrcb and subsample chrominnace channels
    transcol = cv2.cvtColor(img1, cv2.COLOR_RGB2YCR_CB)
    SSV = 2
    SSH = 2
    crf = cv2.boxFilter(transcol[:, :, 1], ddepth=-1, ksize=(2, 2))
    cbf = cv2.boxFilter(transcol[:, :, 2], ddepth=-1, ksize=(2, 2))
    crsub = crf[::SSV, ::SSH]
    cbsub = cbf[::SSV, ::SSH]
    imSub = [transcol[:, :, 0], crsub, cbsub]

    TransAll = []
    TransAllQuant = []

    ch = ['Y', 'Cr', 'Cb']
    for idx, channel in enumerate(imSub):
        channelrows = channel.shape[0]
        channelcols = channel.shape[1]
        Trans = np.zeros((channelrows, channelcols), np.float32)
        TransQuant = np.zeros((channelrows, channelcols), np.float32)
        blocksV = channelrows / B
        blocksH = channelcols / B
        vis0 = np.zeros((channelrows, channelcols), np.float32)
        vis0[:channelrows, :channelcols] = channel
        vis0 = vis0 - 128
        for row in range(int(blocksV)):
            for col in range(int(blocksH)):
                currentblock = cv2.dct(vis0[row * B:(row + 1) * B, col * B:(col + 1) * B])
                Trans[row * B:(row + 1) * B, col * B:(col + 1) * B] = currentblock
                TransQuant[row * B:(row + 1) * B, col * B:(col + 1) * B] = np.round(currentblock / Q[idx])
        TransAll.append(Trans)
        TransAllQuant.append(TransQuant)
    return TransAllQuant, h, w


def jpgInverseDCT(coeffs, h, w):
    DecAll = np.zeros((int(h), int(w), 3), np.uint8)
    for idx, channel in enumerate(coeffs):
        channelrows = channel.shape[0]
        channelcols = channel.shape[1]
        blocksV = channelrows / B
        blocksH = channelcols / B
        back0 = np.zeros((channelrows, channelcols), np.uint8)
        for row in range(int(blocksV)):
            for col in range(int(blocksH)):
                dequantblock = channel[row * B:(row + 1) * B, col * B:(col + 1) * B] * Q[idx]
                currentblock = np.round(cv2.idct(dequantblock)) + 128
                currentblock[currentblock > 255] = 255
                currentblock[currentblock < 0] = 0
                back0[row * B:(row + 1) * B, col * B:(col + 1) * B] = currentblock
        back1 = cv2.resize(back0, (int(w), int(h)))
        DecAll[:, :, idx] = np.round(back1)

    return cv2.cvtColor(DecAll, cv2.COLOR_YCR_CB2RGB)


