from Attacker import attackerAgainstNoninvertibleEmbedder
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

s = Image.open('wm_picture.jpg')

fake_s, fake_c, fake_w = attackerAgainstNoninvertibleEmbedder(s, embed_type='normal')
