from NoninvertibleEmbedder import nonInvertibleEmbedder
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import Detector
import store_load as sl
import os

l = 100
sameSeed = 1
role = 'detector'
img_category = 'high_contrast_'
# open stegowork, coverwork and watermark
img_paths = sl.get_imagepaths_by_name(img_category)[0]
# TODO: loop over img_paths and select from that list
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
detect_result = Detector.detect(w, s, c, sameSeed)
# STORE DATA
sl.save_data(detect_result, img_paths, role, 'orig_' + str(sameSeed))

print("test against true author:")
print(detect_result)
load_result_path = sl.get_datapaths_by_name('orig_' + str(sameSeed), role, img_name)
print(sl.load_data(load_result_path[0]))

# LOAD DATA
# open fake original and fake watermark
# we need to select the first item because it is still a list eventhough it has
# only one item
data_paths = sl.get_datapaths_by_name('fake_wm', 'attacker', img_name)[0]
w = np.load(data_paths)
# we need to select the first item because it is still a list eventhough it has
# only one item
data_paths = sl.get_datapaths_by_name('fake_wm_img', 'attacker', img_name)[0]
c2 = Image.open(data_paths)
detect_result = Detector.detect(w, s, c2, sameSeed)
# STORE DATA
sl.save_data(detect_result, img_paths, role, 'fake_' + str(sameSeed))

print("test against fake author:")
print(detect_result)
load_result_path = sl.get_datapaths_by_name('fake_' + str(sameSeed), role, img_name)
print(sl.load_data(load_result_path[0]))
