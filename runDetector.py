from NoninvertibleEmbedder import nonInvertibleEmbedder
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import Detector

l = 100
# open stegowork, coverwork and watermark
c = Image.open('picture.jpg')
w = np.load('images/wm'+'.npy')
s = Image.open('images/wm_picture''.jpg')
print("test against true author:")
print(Detector.detect(w, s, c))

# open fake original and fake watermark
c2 = Image.open('images/fake_picture'+'.jpg')
w = np.load('images/fake_wm'+'.npy')
print("test against fake author:")
print(Detector.detect(w, s, c2))
