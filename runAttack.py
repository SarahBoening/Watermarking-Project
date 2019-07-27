from Attacker import attackerAgainstNoninvertibleEmbedder
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

# initialize embedding type
embed_type = 'normal'
# open image
s = Image.open('wm_picture_'+embed_type+'.jpg')
# generate fake original, watermark
fake_s, fake_c, fake_w = attackerAgainstNoninvertibleEmbedder(s, embed_type)
# convert nparray to image
fake_c_img = Image.fromarray(fake_c.astype('uint8'), mode='RGB')
# fake_c_img.show()
# save image
fake_c_img.save('fake_picture_'+embed_type+'.jpg')
