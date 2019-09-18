import numpy as np
from PIL import Image
import Detector
import store_load as sl
import os

l = 100
sameSeed = 1
alpha = 0.04
role = 'detector'
img_category = ''
# open stegowork, coverwork and watermark
image_paths = sl.get_imagepaths_by_name(img_category)
for img_paths in image_paths:
    print("Working on file: " + img_paths)
    c = Image.open(img_paths)
    # extract name of image to investigate from current path
    img_name = os.path.basename(img_paths)
    # remove endings
    img_name = img_name.split('.')[0]
    # LOAD DATA
    # we need to select the first item because it is still a list eventhough it has
    # only one item
    data_paths = sl.get_datapaths_by_name('wm', 'embedder', img_name)[0]
    w = np.load(data_paths)
    # we need to select the first item because it is still a list eventhough it has
    # only one item
    data_paths = sl.get_datapaths_by_name('wm_img', 'embedder', img_name)[0]
    s = Image.open(data_paths)
    detect_result = Detector.detect(w, s, c, alpha, sameSeed)
    # STORE DATA
    sl.save_data(detect_result, img_paths, role, 'orig_' + str(sameSeed))

    print("test against true author:" + img_name)
    print(detect_result)

    # LOAD DATA
    # open fake original and fake watermark
    # we need to select the first item because it is still a list eventhough it has
    # only one item
    data_paths = sl.get_datapaths_by_name('fake_wm', 'attacker', img_name)[0]
    w = np.load(data_paths)
    # we need to select the first item because it is still a list even though it has
    # only one item
    data_paths = sl.get_datapaths_by_name('fake_wm_img', 'attacker', img_name)[0]
    c2 = Image.open(data_paths)
    detect_result = Detector.detect(w, s, c2, alpha, sameSeed)
    # STORE DATA
    sl.save_data(detect_result, img_paths, role, 'fake_' + str(sameSeed))

    print("test against fake author:" + img_name)
    print(detect_result)
