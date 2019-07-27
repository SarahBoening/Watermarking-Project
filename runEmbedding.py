from NoninvertibleEmbedder import noninvertibleEmbedder
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

# initialize embedding type
embed_type = 'BBS'
# open image (default in RGB)
img = Image.open('picture.jpg')
# img.show()
# length of the embeded string
l = 100
# watermark drawn independently from N(0, 1)-distribution
w = np.random.normal(0.0, 1.0, (1,l)) # e.g [[ ... ]]
# print(w)
# use embedder
s = noninvertibleEmbedder(w, img, embed_type)
# convert nparray to image
wm_img = Image.fromarray(s.astype('uint8'), mode='RGB')
# wm_img.show()
# save image and watermark
wm_img.save(embed_type+'_wm_picture''.jpg')
np.save(embed_type+'_wm',w)
