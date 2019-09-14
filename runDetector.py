from NoninvertibleEmbedder import nonInvertibleEmbedder
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import Detector

l = 100
# open stegowork, coverwork and watermark
c = Image.open('images/originals/high_contrast_01.jpg')
w = np.load('Analysis_data/wm'+'.npy')
s = Image.open('images/wm_picture_new''.jpg')
print("test against true author:")
print(Detector.detect(w, s, c, sameSeed=1))

# open fake original and fake watermark
c2 = Image.open('images/fake_picture_new'+'.jpg')
w = np.load('Analysis_data/fake_wm'+'.npy')
print("test against fake author:")
print(Detector.detect(w, s, c2, sameSeed=1))
