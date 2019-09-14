from NoninvertibleEmbedder import nonInvertibleEmbedder
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
from PIL import Image
import store_load as sl

print('run embedder')
# open image (default in RGB)
img_paths = sl.get_imagepaths_by_name('high_contrast_')
img_path = img_paths[0]
img = Image.open(img_path)
# STORE DATA
# save image matrix
sl.save_data(np.array(img), img_path, 'img_matrix')

# img.show()
# length of the embedded string
l = 100
# watermark drawn independently from N(0, 1)-distribution
w = np.random.normal(0.0, 1.0, (1, l))
# print(w)
# use embedder
s = nonInvertibleEmbedder(w, img, img_path)
# STORE DATA
# save watermarked image matrix
sl.save_data(s, img_path, 'wm_img_matrix')
# STORE DATA
# convert nparray to image and save it
sl.save_nparray_as_img(s, img_path, 'wm_img')
# STORE DATA
# save watermark
sl.save_data(w, img_path, 'wm')
