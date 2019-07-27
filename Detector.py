"""
Detector to verify ownership.
Extracts a watermark from a stegowork and compares it with a given watermark
to compute a similarity value. Compares similiraty value with a threshold value Tau
to decide if the extracted and given watermark are the same.
"""

import math as m

def detect(given_wm, stegowork, coverwork, p, q):
    """
    Execute detection pipeline.

    given_wm: list containing the watermark to check against
    stegowork: multi-dimensional list, watermarked image
    coverwork: multi-dimensional list, original coverwork without watermark
    p, q: integers, primes used to calculate a random path

    returns True or False - given_wm is present in the stegowork.
    """
    TAU = 0.7

    path = BBS(p, q) # used in the extractor functions?
    stego_features = extract_features(stegowork) # probably replace with modified_DCT()
    cover_features = extract_features(coverwork) # probably replace with modified_DCT()
    extracted_wm = extract_watermark(stego_features, cover_features)
    similarity = calc_similarity(given_wm, extracted_wm)
    if similarity > TAU:
        return True
    else:
        return False

def extract_features(image):
    """
    Use a modified DCT to extract features from an image.

    image: multi-dimensional list, an image

    returns list of tuples, list contains feature positions
    """
    # features = modified_DCT(coverwork)
    # return features
    pass

def extract_watermark(stego_features, cover_features):
    """
    Extract watermark using the inversed embedding function.

    stego_features: list of tuples, list contains feature positions
    cover_features: list of tuples, list contains feature positions

    returns list, contains the extracted watermark
    """
    watermark = inverse_modified_DCT(stego_features, cover_features)
    pass

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