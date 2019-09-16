import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import store_load as sl
from pprint import pprint as pp

if __name__=="__main__":
    print("plotting data")
    
    data_set_paths = sl.get_datapaths_by_name('DCTCoeffs', 'embedder', 'high_contrast_')
    for path in data_set_paths:
        blue_channel_DCTs = sl.get_blue_channel(path)
        # print(blue_channel_DCTs)

    data_set_paths = sl.get_datapaths_by_name('wm', 'embedder', 'high_contrast_')
    for path in data_set_paths:
        wm_data = sl.load_data(path)[0]  # select first element to remove unnecessary outer list
        # print(wm_data)

    data_set_paths = sl.get_datapaths_by_name('wm_img_matrix', 'embedder', 'high_contrast_')
    for path in data_set_paths:
        wm_img_matrix_data = sl.get_blue_channel(path)
        # print(wm_img_matrix_data)

    data_set_paths = sl.get_datapaths_by_name('wm_img', 'embedder', 'high_contrast_')
    for path in data_set_paths:
        img = np.array(Image.open(path))
        # next line actually does the same as "sl.get_blue_channel()"
        wm_img_data = [img[idx_h][0][2] for idx_h, value in enumerate(img)]
        # print(wm_img_data)

    plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':100})
    plt.hist(blue_channel_DCTs, bins=10)
    plt.gca().set(title='DCTs, blue color channel')
    plt.show()
