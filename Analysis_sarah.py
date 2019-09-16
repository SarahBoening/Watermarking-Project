import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import store_load as sl

if __name__=="__main__":
    # image_paths = sl.get_imagepaths_by_name(img_category)
    data_set_paths = sl.get_datapaths_by_name('DCTCoeffs', 'embedder', '')
    i = 0
    av = 0
    for path in data_set_paths:
        blue_channel_DCTs = sl.get_blue_channel(path)
        av += np.average(blue_channel_DCTs)
        i += 1
    print(av/i)

