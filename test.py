from Statistics import *
from PIL import Image
import numpy as np

img = np.array(Image.open('images/originals/high_contrast_01.jpg'))
print(img.shape)
im_b = img[:, :, 2]
vals = subhistcount(im_b, 0, 255)
print(vals.shape)
print(chisquaretestcorr(vals))
print(chisquaretestuncorr(vals))
print(vals)
