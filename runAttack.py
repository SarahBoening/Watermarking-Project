from Attacker import attackerAgainstNoninvertibleEmbedder
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

# initialize embedding type
embed_type = 'BBS'
# open image
s = Image.open(embed_type+'_wm_picture'+'.jpg')
# generate fake original, watermark
fake_s, fake_c, fake_w = attackerAgainstNoninvertibleEmbedder(s, embed_type)
# convert nparray to image
fake_c_img = Image.fromarray(fake_c.astype('uint8'), mode='RGB')
fake_s_img = Image.fromarray(fake_s.astype('uint8'), mode='RGB')
# save image and watermark
fake_c_img.save(embed_type+'_fake_picture'+'.jpg')
fake_s_img.save(embed_type+'_fake_wm_picture'+'.jpg')
np.save(embed_type+'_fake_wm',fake_w)
