from Attacker import attackerAgainstNoninvertibleEmbedder
from PIL import Image
import store_load as sl

print('run attacker')
role = 'attacker'
img_category = ''
# initialize l
l = 100
sameSeed = False
alpha = 0.04
# LOAD DATA
# open image
wm_orig_images = sl.get_datapaths_by_name('wm_img', 'embedder', img_category)

for wm_orig_img in wm_orig_images:
    print("Working on file: " + wm_orig_img)
    s = Image.open(wm_orig_img)
    # generate fake original, watermark
    fake_s, fake_c, fake_w = attackerAgainstNoninvertibleEmbedder(
        s, l, wm_orig_img, sameSeed)
    # STORE DATA
    sl.save_data(fake_c, wm_orig_img, role, 'fake_cw') # cw = coverwork
    sl.save_data(fake_s, wm_orig_img, role, 'fake_s')
    sl.save_data(fake_w, wm_orig_img, role, 'fake_wm')
    # convert nparray to image and save it
    sl.save_nparray_as_img(fake_c, wm_orig_img, role, 'fake_cw_img') # cw = coverwork
    # sl.save_nparray_as_img(fake_s, wm_orig_img, role, 'fake_s_img')
    sl.save_nparray_as_img(fake_s, wm_orig_img, role, 'fake_wm_img')
