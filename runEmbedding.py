from NoninvertibleEmbedder import nonInvertibleEmbedder
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

# open image (default in RGB)
img = Image.open('bild.jpg')
# img.show()
# length of the embedded string
l = 100
# watermark drawn independently from N(0, 1)-distribution
w = np.random.normal(0.0, 1.0, (1, l))
# print(w)
# use embedder
s = nonInvertibleEmbedder(w, img)
# convert nparray to image
wm_img = Image.fromarray(s.astype('uint8'), mode='RGB')
# wm_img.show()
# save image and watermark
wm_img.save('images/wm_picture_new.jpg')
np.save('images/wm', w)
