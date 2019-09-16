from NoninvertibleEmbedder import nonInvertibleEmbedder
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
from PIL import Image
import store_load as sl

print('run embedder')
role = 'embedder'
img_category = ''
# open image (default in RGB)
image_paths = sl.get_imagepaths_by_name(img_category)
# TODO: loop over img_paths
for img_path in image_paths:
    print("Working on file: " + img_path)
    img = Image.open(img_path)
    # STORE DATA
    # save image matrix
    sl.save_data(np.array(img), img_path, role, 'img_matrix')

    # img.show()
    # length of the embedded string
    l = 100
    # watermark drawn independently from N(0, 1)-distribution
    w = np.random.normal(0.0, 1.0, (1, l))
    # print(w)
    # use embedder
    s = nonInvertibleEmbedder(w, img, img_path, role)
    # STORE DATA
    # save watermarked image matrix
    sl.save_data(s, img_path, role, 'wm_img_matrix')
    # STORE DATA
    # convert nparray to image and save it
    sl.save_nparray_as_img(s, img_path, role, 'wm_img')
    # STORE DATA
    # save watermark
    sl.save_data(w, img_path, role, 'wm')
