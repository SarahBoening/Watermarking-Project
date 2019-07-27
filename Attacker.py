import numpy as np
import NoninvertableEmbedder as nie
import YCrCbDCT as dct


def attackerAgainstNoninvertibleEmbedder(s, embed_type='normal'):
    # 1. DCT of s
    S, x, y = dct.jpgDCT(s)
    # 2. Variations: take l most significant bits / bbs path / upper left corner block values
    l = 10
    # 3. generate 2 vectors u, v with random values e N(0,1)
    u = np.random.uniform(0.0, 1.0, (1, l))
    v = np.random.uniform(0.0, 1.0, (1, l))
    # 4. inverse embedding with u
    # 5. inverse embedding with v
    # 6. replace coefficients
    # 7. inverse transform

    # 8. compute fake original
    fake_c = np.zeros((x, y, 3))
    # 9. hash bit string of length l
    b = nie.hashimage(fake_c, l)
    # 10. compute fake watermark
    fake_w = []
    # 11. generate fake watermarked work with embedder
    fake_s = nie.noninvertibleEmbedder(fake_c, fake_w)

    return fake_s, fake_c, fake_w

