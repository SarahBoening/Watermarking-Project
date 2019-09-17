import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import store_load as sl
from pprint import pprint as pp
import cv2

if __name__=="__main__":
    """
    # average of DCT coefficient before embedding
    data_set_paths = sl.get_datapaths_by_name('DCTCoeffs', 'embedder', '')
    averages_before = {}
    for path in data_set_paths:
        blue_channel_DCTs = np.load(path)[:,:,2]
        name = path.split("/")[-1]
        averages_before[name] = np.average(blue_channel_DCTs)

    # average of DCT coefficient after embedding
    data_set_paths = sl.get_datapaths_by_name('DCTCoeffs_after', 'embedder', '')
    averages_after = {}
    for path in data_set_paths:
        blue_channel_DCTs = np.load(path)[:,:,2]
        name = path.split("/")[-1]
        averages_after[name] = np.average(blue_channel_DCTs)

    for key in sorted(averages_before.keys()):
        pp(key)
        pp(averages_before[key])
        pp(averages_after[key])
    """

    """
    # average of upperleft corner DCT coefficient before embedding
    data_set_paths = sl.get_datapaths_by_name('DCTCoeffs', 'embedder', '')
    averages_before = {}
    for path in data_set_paths:
        blue_channel_DCTs = np.load(path)[:,:,2]
        upperleft_DCTs = blue_channel_DCTs[0::8,0::8]
        name = path.split("/")[-1]
        averages_before[name] = np.average(upperleft_DCTs)

    # average of upperleft corner DCT coefficient after embedding
    data_set_paths = sl.get_datapaths_by_name('DCTCoeffs_after', 'embedder', '')
    averages_after = {}
    for path in data_set_paths:
        blue_channel_DCTs = np.load(path)[:,:,2]
        upperleft_DCTs = blue_channel_DCTs[0::8,0::8]
        name = path.split("/")[-1]
        averages_after[name] = np.average(upperleft_DCTs)

    for key in sorted(averages_before.keys()):
        pp(key)
        pp(averages_before[key])
        pp(averages_after[key])
    """

    """
    # pixels of image original
    data_set_paths = sl.get_datapaths_by_name('img_matrix', 'embedder', '')
    pixels_original = {}
    for path in data_set_paths:
        im = np.load(path)
        YCbCr = cv2.cvtColor(im.astype('float32'), cv2.COLOR_RGB2YCR_CB)
        blue_channel_DCTs = sl.get_blue_channel(path)
        name = path.split("/")[-1]
        pixels_original[name] = blue_channel_DCTs

    # pixels of fake image
    data_set_paths = sl.get_datapaths_by_name('fake_s', 'attacker', '')
    pixels_fake = {}
    for path in data_set_paths:
        im = np.load(path)
        YCbCr = cv2.cvtColor(im.astype('float32'), cv2.COLOR_RGB2YCR_CB)
        blue_channel_DCTs = sl.get_blue_channel(path)
        name = path.split("/")[-1]
        pixels_fake[name] = blue_channel_DCTs

    for key in sorted(pixels_fake):
        pp(key)
        pp(np.sum(np.abs(np.subtract(pixels_original[key],pixels_fake[key]))))
    """

    # pixels of image original
    data_set_paths = sl.get_datapaths_by_name('DCTCoeffs', 'embedder', '')
    DCTs_original = {}
    for path in data_set_paths:
        blue_channel_DCTs = np.load(path)[:,:,2]
        upperleft_DCTs = blue_channel_DCTs[0::8,0::8]
        name = path.split("/")[-1]
        DCTs_original[name] = upperleft_DCTs

    # pixels of fake image
    data_set_paths = sl.get_datapaths_by_name('DCTCoeffs', 'attacker', '')
    DCTs_fake = {}
    for path in data_set_paths:
        blue_channel_DCTs = np.load(path)[:,:,2]
        upperleft_DCTs = blue_channel_DCTs[0::8,0::8]
        name = path.split("/")[-1]
        DCTs_fake[name] = upperleft_DCTs

    for key in sorted(DCTs_original):
        pp(key)
        pp(np.sum(np.equal(DCTs_original[key],DCTs_fake[key]).astype(int)))
        pp(np.sum(np.abs(np.subtract(DCTs_original[key],DCTs_fake[key]))))
