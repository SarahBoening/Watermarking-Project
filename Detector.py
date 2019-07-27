"""
Detector to verify ownership.
Extracts a watermark from a stegowork and compares it with a given watermark
to compute a similarity value. Compares similiraty value with a threshold value Tau
to decide if the extracted and given watermark are the same.
"""

def detect(given_wm, stegowork, coverwork, p, q):
    """
    Execute detection pipeline.

    given_wm: list containing the watermark to check against
    stegowork: multi-dimensional list, watermarked image
    coverwork: multi-dimensional list, original coverwork without watermark
    p, q: integers, primes used to calculate a random path

    returns True or False - given_wm is present in the stegowork.
    """

    features = extract_features(coverwork)

def extract_features(coverwork):
    """
    Use a modified DCT to extract features from coverwork.

    coverwork: multi-dimensional list, original coverwork without watermark

    returns list of tuples, list contains feature positions
    """
    # features = modified_DCT(coverwork)
    # return features
    pass