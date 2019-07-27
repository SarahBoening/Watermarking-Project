from NoninvertibleEmbedder import noninvertibleEmbedder
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import Detector

l = 100
embed_type = 'BBS'

# open stegowork, coverwork and watermark
c = Image.open('picture.jpg')
w = np.load(embed_type+'_wm'+'.npy')
s = Image.open(embed_type+'_wm_picture''.jpg')
print("test against true author:")
print(Detector.detect(w, s, c, embed_type=embed_type))

# open fake original and fake watermark
c2 = Image.open(embed_type+'_fake_picture'+'.jpg')
w = np.load(embed_type+'_fake_wm'+'.npy')
print("test against fake author:")
print(Detector.detect(w, s, c2, embed_type=embed_type))
