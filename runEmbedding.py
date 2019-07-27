from NoninvertibleEmbedder import noninvertibleEmbedder
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

# open image (default in RGB)
img = Image.open('picture.jpg')
# img.show()
# length of the embeded string
l = 100
# watermark drawn independently from N(0, 1)-distribution
w = np.random.normal(0.0, 1.0, (1,l)) # e.g [[ ... ]]
# print(w)
# use embedder
s = noninvertibleEmbedder(w, img, embed_type='DCT')
# open image in RGB
wm_img = Image.fromarray(s.astype('uint8'), mode='RGB')
wm_img.show()
# save image
wm_img.save('wm_picture.jpg')
