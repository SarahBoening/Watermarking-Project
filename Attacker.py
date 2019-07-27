import numpy as np
import YCrCbDCT as dct
from NoninvertibleEmbedder import *
from PIL import Image

def attackerAgainstNoninvertibleEmbedder(s, embed_type='normal'):
    # step 1: perform DCT of watermarked image
    S, x, y = dct.jpgDCT(s)
    # initialize l
    l = 100
    # step 2: generate 2 vectors u, v with random values e N(0,1)
    u = np.random.uniform(0.0, 1.0, (1, l))
    v = np.random.uniform(0.0, 1.0, (1, l))
    # step 3: invert embedding
    # inverse embedding with u
    c1 = invertEmbedding(S, u, np.ones(l), l, x, y, embed_type)
    # inverse embedding with v
    c2 = invertEmbedding(S, v, np.zeros(l), l, x, y, embed_type)
    # step 4: compute fake original by averaging
    fake_c = np.array((c1 + c2) / 2)
    # print("hello1","s",s,"c1",c1,"c2",c2,"fake",fake_c)
    fake_c_img = Image.fromarray(fake_c.astype('uint8'), mode='RGB')
    # step 5: hash bit string of length l
    b = hashimage(fake_c_img, l)
    # step 6: compute fake watermark
    fake_w = np.array([np.zeros(l)])
    for i in range(0, l):
        if b[i] == 1:
            fake_w[0][i] = u[0][i]
        elif b[i] == 0:
            fake_w[0][i] = v[0][i]
    # step 7: generate fake watermarked work with embedder
    fake_s = noninvertibleEmbedder(fake_w, fake_c_img)

    return fake_s, fake_c, fake_w
