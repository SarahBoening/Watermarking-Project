from NoninvertibleEmbedder import noninvertibleEmbedder
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

# img = mpimg.imread('picture.jpg')
img = Image.open('picture.jpg')
#print(np.array(img)[:10, :10, 0])
#img.show()
l = 100
w = np.random.uniform(0.0, 1.0, (1,l))
print(w)
s = noninvertibleEmbedder(w, img, embed_type='DCT')
#s = noninvertibleEmbedder(w, img)
image = Image.fromarray(s.astype('uint8'), 'RGB')
image.show()
# save image
image.save('wm_picture.jpg')
# print(img.shape)
#plt.imshow(s)
#plt.show()
