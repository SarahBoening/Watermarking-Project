"""
Detector to verify ownership.
Extracts a watermark from a stegowork and compares it with a given watermark
to compute a similarity value. Compares similiraty value with a threshold value Tau
to decide if the extracted and given watermark are the same.
"""

import math as m
import numpy as np
from pprint import pprint as pp

import BlumBlumShup as BBS
from NoninvertibleEmbedder import hashimage
from YCrCbDCT import jpgDCT

def detect(given_wm, stegowork, coverwork, TAU = 0.7): # TAU is randomly chosen
    """
    Execute detection pipeline.

    given_wm: list containing the watermark to check against
    stegowork: multi-dimensional list, watermarked image
    coverwork: multi-dimensional list, original coverwork without watermark
    TAU: float, threshold value to decide if given_wm is present in the stegowork.

    returns True or False - given_wm is present in the stegowork.
    """

    path = hashimage(coverwork, given_wm.shape[1]) # used in the extractor functions?
    stego_coeffs, x, y = jpgDCT(stegowork) # probably replace with modified_DCT()
    cover_coeffs, x, y = jpgDCT(coverwork) # probably replace with modified_DCT()
    extracted_wm = inverse_modified_DCT(cover_coeffs,stego_coeffs,  path, given_wm.shape[1])
    # print(np.asarray(extracted_wm))
    # print(given_wm)
    for i in range(len(given_wm)):
        print(abs(np.asarray(extracted_wm)[i]) - abs(given_wm[i]))

    # similarity = calc_similarity(given_wm, extracted_wm)
#     if similarity > TAU:
#         return True
#     else:
#         return False

def inverse_modified_DCT(sw_coeffs, cw_coeffs, path, l, alpha = 0.025):
    watermark = list()
    i = 0
    stop = False
    for j in range(8, sw_coeffs.shape[0] - 8, 8):
        for k in range(8, sw_coeffs.shape[1] - 8, 8):
            if path[i] == 1:
                w = sw_coeffs[j,k,2]/(cw_coeffs[j,k,2] * alpha) - 1/ alpha
                watermark.append(w)
            else:
                w = 1/alpha - sw_coeffs[j,k,2]/(cw_coeffs[j,k,2] * alpha)
                watermark.append(w)
            i += 1
            # print(f"sw: {sw_coeffs[j,k,2]}")
            # print(f"cw: {cw_coeffs[j,k,2]}")
            # print(f"w: {w}")
            if i >= l:
                stop = True
                break
        if stop:
            break
    
    return watermark

def calc_similarity(given_wm, extracted_wm):
    """
    Calculates a similarity value for given_wm and extracted_wm.

    given_wm: list, contains the given watermark
    extracted_wm: list, contains the extracted watermark

    returns: float, calculated similarity value
    """

    numerator = extracted_wm * given_wm
    denominator = m.sqrt(extracted_wm * extracted_wm) # basic spread spectrum scheme
    # denominator = m.sqrt((extracted_wm * extracted_wm) * (given_wm * given_wm))
    similarity = numerator / denominator
    return similarity