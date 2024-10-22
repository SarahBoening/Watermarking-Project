"""
Detector to verify ownership.
Extracts a watermark from a stegowork and compares it with a given watermark
to compute a similarity value. Compares similiraty value with a threshold value Tau
to decide if the extracted and given watermark are the same.
"""

import math as m
import numpy as np
import BlumBlumShup as bbs
from NoninvertibleEmbedder import hashimage
from YCrCbDCT import jpgDCT


def detect(given_wm, stegowork, coverwork, alpha, sameSeed, TAU=0.3):  # TAU is randomly chosen
    """
    Execute detection pipeline.

    given_wm: list containing the watermark to check against
    stegowork: multi-dimensional list, watermarked image
    coverwork: multi-dimensional list, original coverwork without watermark
    TAU: float, threshold value to decide if given_wm is present in the stegowork.

    returns similarity and True or False - given_wm is present in the stegowork.
    """

    # used in the extractor functions
    b = hashimage(coverwork, given_wm.shape[1])
    # probably replace with modified_DCT()
    stego_coeffs, x, y = jpgDCT(stegowork)
    # probably replace with modified_DCT()
    cover_coeffs, x, y = jpgDCT(coverwork)
    # Extract watermark from stego and cover
    extracted_wm = inverse_modified_DCT(
        cover_coeffs, stego_coeffs, b, given_wm.shape[1], alpha, sameSeed)
    # calculate similarity between given and extracted watermark
    similarity = calc_similarity(np.asarray(extracted_wm), given_wm[0])
    if abs(similarity) > TAU:
        return similarity, True
    else:
        return similarity, False


def inverse_modified_DCT(sw_coeffs, cw_coeffs, b, l, alpha=0.04, sameSeed=1):
    """
    Extract watermark from stego and cover.

    sw_coeffs: DCT coefficients of stegowork
    cw_coeffs: DCT coefficients of coverwork
    b: hash string from coverwork
    l: length of watermark
    returns: extracted watermark
    """
    watermark = list()

    temp = np.array(sw_coeffs)
    # same as embedder
    if sameSeed == 1:
        p = 5999
        q = 60107
        Mb = p * q
        xi = 20151208
    # same as attacker
    elif sameSeed == 2:
        p = 9539
        q = 54193
        Mb = p * q
        xi = 83739
    # new seed
    else:
        p = 9539
        q = 54193
        Mb = p * q
        xi = 94574

    path = bbs.getDCTBBSPath(l, xi, Mb, temp.shape[1]-8, temp.shape[0]-8)
    for p in range(0, path.shape[1]):
        # get the next block position to embed
        n = path[0][p]
        m = path[1][p]
        w = 0
        # currently also embed at Cb color channel (dimension 2)
        # calculate watermark based on noninvertible algorithm
        if b[p] == 1:
            w = sw_coeffs[m, n, 2] / (cw_coeffs[m, n, 2] * alpha) - 1 / alpha
        elif b[p] == 0:
            w = 1 / alpha - sw_coeffs[m, n, 2] / (cw_coeffs[m, n, 2] * alpha)
        watermark.append(w)
    return watermark


def calc_similarity(given_wm, extracted_wm):
    """
    Calculates a similarity value for given_wm and extracted_wm.

    given_wm: list, contains the given watermark
    extracted_wm: list, contains the extracted watermark

    returns: float, calculated similarity value
    """

    numerator = np.dot(extracted_wm, given_wm)
    # denominator = m.sqrt(np.dot(extracted_wm, extracted_wm))  # basic spread spectrum scheme
    denominator = m.sqrt(
        np.dot(extracted_wm, extracted_wm) * np.dot(given_wm, given_wm)
    )

    similarity = numerator / denominator
    return similarity
