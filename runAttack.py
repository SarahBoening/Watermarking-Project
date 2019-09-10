from Attacker import attackerAgainstNoninvertibleEmbedder
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

# TODO: Perform several attacks (single and multiple public watermarked images)
# initialize l
l = 100
# open image
s = Image.open('images/wm_picture'+'.jpg')
# generate fake original, watermark
fake_s, fake_c, fake_w = attackerAgainstNoninvertibleEmbedder(s, l)
# convert nparray to image
fake_c_img = Image.fromarray(fake_c.astype('uint8'), mode='RGB')
fake_s_img = Image.fromarray(fake_s.astype('uint8'), mode='RGB')
# save image and watermark
fake_c_img.save('images/fake_picture'+'.jpg')
fake_s_img.save('images/fake_wm_picture'+'.jpg')
np.save('images/fake_wm', fake_w)
