import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image
import store_load as sl
from pprint import pprint as pp
import cv2
import matplotlib.cm as cm

def add_text(ax,data):
    # Loop over data dimensions and create text annotations.
    for i in range(8):
        for j in range(8):
            text = ax.text(j, i, int(data[i, j]), fontsize=6.5,
                           ha="center", va="center", color="w")
if __name__=="__main__":
    """
    # average of DCT coefficient before embedding
    data_set_paths = sl.get_datapaths_by_name('DCTCoeffs', 'embedder', '')
    averages_before = {}
    for path in data_set_paths:
        blue_channel_DCTs = np.load(path)[:,:,2]
        name = path.split("/")[-1]
        averages_before[name] = np.average(blue_channel_DCTs)

    # average of DCT coefficient after embedding
    data_set_paths = sl.get_datapaths_by_name('DCTCoeffs_after', 'embedder', '')
    averages_after = {}
    for path in data_set_paths:
        blue_channel_DCTs = np.load(path)[:,:,2]
        name = path.split("/")[-1]
        averages_after[name] = np.average(blue_channel_DCTs)

    for key in sorted(averages_before.keys()):
        pp(key)
        pp(averages_before[key])
        pp(averages_after[key])
    """

    """
    # average of upperleft corner DCT coefficient before embedding
    data_set_paths = sl.get_datapaths_by_name('DCTCoeffs', 'embedder', '')
    averages_before = {}
    for path in data_set_paths:
        blue_channel_DCTs = np.load(path)[:,:,2]
        upperleft_DCTs = blue_channel_DCTs[0::8,0::8]
        name = path.split("/")[-1]
        averages_before[name] = np.average(upperleft_DCTs)

    # average of upperleft corner DCT coefficient after embedding
    data_set_paths = sl.get_datapaths_by_name('DCTCoeffs_after', 'embedder', '')
    averages_after = {}
    for path in data_set_paths:
        blue_channel_DCTs = np.load(path)[:,:,2]
        upperleft_DCTs = blue_channel_DCTs[0::8,0::8]
        name = path.split("/")[-1]
        averages_after[name] = np.average(upperleft_DCTs)

    for key in sorted(averages_before.keys()):
        pp(key)
        pp(averages_before[key])
        pp(averages_after[key])
    """

    """
    # pixels of image original
    data_set_paths = sl.get_datapaths_by_name('img_matrix', 'embedder', '')
    pixels_original = {}
    for path in data_set_paths:
        im = np.load(path)
        YCbCr = cv2.cvtColor(im.astype('float32'), cv2.COLOR_RGB2YCR_CB)
        blue_channel_DCTs = sl.get_blue_channel(path)
        name = path.split("/")[-1]
        pixels_original[name] = blue_channel_DCTs

    # pixels of fake image
    data_set_paths = sl.get_datapaths_by_name('fake_cw', 'attacker', '')
    pixels_fake = {}
    for path in data_set_paths:
        im = np.load(path)
        YCbCr = cv2.cvtColor(im.astype('float32'), cv2.COLOR_RGB2YCR_CB)
        blue_channel_DCTs = sl.get_blue_channel(path)
        name = path.split("/")[-1]
        pixels_fake[name] = blue_channel_DCTs

    for key in sorted(pixels_fake):
        pp(key)
        pp(np.sum(np.abs(np.subtract(pixels_original[key],pixels_fake[key]))))
    """

    """
    # pixels of image original
    data_set_paths = sl.get_datapaths_by_name('DCTCoeffs', 'embedder', '')
    DCTs_original = {}
    for path in data_set_paths:
        blue_channel_DCTs = np.load(path)[:,:,2]
        upperleft_DCTs = blue_channel_DCTs[0::8,0::8]
        name = path.split("/")[-1]
        DCTs_original[name] = upperleft_DCTs

    # pixels of fake image
    data_set_paths = sl.get_datapaths_by_name('DCTCoeffs', 'attacker', '')
    DCTs_fake = {}
    for path in data_set_paths:
        blue_channel_DCTs = np.load(path)[:,:,2]
        upperleft_DCTs = blue_channel_DCTs[0::8,0::8]
        name = path.split("/")[-1]
        DCTs_fake[name] = upperleft_DCTs

    for key in sorted(DCTs_original):
        pp(key)
        pp(np.sum(np.equal(DCTs_original[key],DCTs_fake[key]).astype(int)))
        pp(np.sum(np.abs(np.subtract(DCTs_original[key],DCTs_fake[key]))))
    """

    """
    # pixels of image original
    data_set_paths = sl.get_datapaths_by_name('img_matrix', 'embedder', '')
    pixels_original = {}
    for path in data_set_paths:
        im = np.load(path)
        YCbCr = cv2.cvtColor(im.astype('float32'), cv2.COLOR_RGB2YCR_CB)
        name = path.split("/")[-1]
        pixels_original[name] = YCbCr
    # pixels of image original
    data_set_paths = sl.get_datapaths_by_name('wm_img_matrix', 'embedder', '')
    pixels_watermark = {}
    for path in data_set_paths:
        im = np.load(path)
        YCbCr = cv2.cvtColor(im.astype('float32'), cv2.COLOR_RGB2YCR_CB)
        name = path.split("/")[-1]
        pixels_watermark[name] = YCbCr
    # pixels of image fake
    data_set_paths = sl.get_datapaths_by_name('fake_cw', 'attacker', '')
    pixels_fake = {}
    for path in data_set_paths:
        im = np.load(path)
        YCbCr = cv2.cvtColor(im.astype('float32'), cv2.COLOR_RGB2YCR_CB)
        name = path.split("/")[-1]
        pixels_fake[name] = YCbCr
    # pixels of image fake watermarked
    data_set_paths = sl.get_datapaths_by_name('fake_s', 'attacker', '')
    pixels_wm_fake = {}
    for path in data_set_paths:
        im = np.load(path)
        YCbCr = cv2.cvtColor(im.astype('float32'), cv2.COLOR_RGB2YCR_CB)
        name = path.split("/")[-1]
        pixels_wm_fake[name] = YCbCr

    for key in sorted(pixels_wm_fake):
        color = ('y','r','b')
        col1 = ["darkgoldenrod","maroon","navy"]
        col2 = ["goldenrod","coral","royalblue"]
        col3 = ["gold","orangered","blue"]
        col4 = ["darkkhaki","firebrick","mediumslateblue"]
        for i,col in enumerate(color):
            histr_or = cv2.calcHist([pixels_original[key]],[i],None,[256],[0,256])
            histr_wm = cv2.calcHist([pixels_watermark[key]],[i],None,[256],[0,256])
            histr_f = cv2.calcHist([pixels_fake[key]],[i],None,[256],[0,256])
            histr_wmf = cv2.calcHist([pixels_wm_fake[key]],[i],None,[256],[0,256])
            line1 = plt.plot(histr_or,color = col1[i],linestyle="solid",marker="*",label="original image")
            line2 = plt.plot(histr_wm,color = col2[i],linestyle="dashed",marker="<",label="watermarked image")
            line3 = plt.plot(histr_f,color = col3[i],linestyle="dashdot",marker="o",label="fake image")
            line4 = plt.plot(histr_wmf,color = col4[i],linestyle="dotted",marker="x",label="fake watermarked")
            plt.title(key.strip(".npy"))
            plt.legend()
            plt.xlim([0,256])
        plt.show()
    """

    # DCTs of image original
    data_set_paths = sl.get_datapaths_by_name('DCTCoeffs', 'embedder', '')
    DCTs_original = {}
    DCTs_original_dif = {}
    for path in data_set_paths:
      im = np.load(path)[:,:,2]
      name = path.split("/")[-1]
      DCTs_original[name] = im
      result = np.empty((8,8))
      for i in range(0,8):
          for j in range(0,8):
              original = DCTs_original[name][i::8,j::8]
              current = im[i::8,j::8]
              result[i,j] = np.sum(np.abs(np.subtract(original,current)))
      DCTs_original_dif[name] = result

    # DCTs of image watermarked
    data_set_paths = sl.get_datapaths_by_name('DCTCoeffs_after', 'embedder', '')
    DCTs_watermark_dif = {}
    for path in data_set_paths:
      im = np.load(path)[:,:,2]
      name = path.split("/")[-1]
      result = np.empty((8,8))
      for i in range(0,8):
          for j in range(0,8):
              original = DCTs_original[name][i::8,j::8]
              current = im[i::8,j::8]
              result[i,j] = np.sum(np.abs(np.subtract(original,current)))
    #   print(result[0,0])
      DCTs_watermark_dif[name] = result

    # DCTs of image fake
    data_set_paths = sl.get_datapaths_by_name('DCTCoeffs', 'attacker', '')
    DCTs_fake_dif = {}
    for path in data_set_paths:
      im = np.load(path)[:,:,2]
      name = path.split("/")[-1]
      result = np.empty((8,8))
      for i in range(0,8):
          for j in range(0,8):
              original = DCTs_original[name][i::8,j::8]
              current = im[i::8,j::8]
              result[i,j] = np.sum(np.abs(np.subtract(original,current)))
    #   print(result[0,0])
      DCTs_fake_dif[name] = result

    # DCTs of image fake watermarked
    data_set_paths = sl.get_datapaths_by_name('DCTCoeffs_after', 'attacker', '')
    DCTs_wm_fake_dif = {}
    for path in data_set_paths:
      im = np.load(path)[:,:,2]
      name = path.split("/")[-1]
      result = np.empty((8,8))
      for i in range(0,8):
          for j in range(0,8):
              original = DCTs_original[name][i::8,j::8]
              current = im[i::8,j::8]
              result[i,j] = np.sum(np.abs(np.subtract(original,current)))
    #   print(result[0,0])
      DCTs_wm_fake_dif[name] = result

    for i, key in enumerate(sorted(DCTs_wm_fake_dif)):
        fig, axs = plt.subplots(2, 2)
        img1 = axs[0, 0].imshow(DCTs_original_dif[key],cmap=cm.jet,interpolation='nearest')
        axs[0, 0].set_title('Original image')
        add_text(axs[0, 0], DCTs_original_dif[key])
        img2 = axs[0, 1].imshow(DCTs_watermark_dif[key],cmap=cm.jet,interpolation='nearest')
        axs[0, 1].set_title('Watermarked image')
        add_text(axs[0, 1], DCTs_watermark_dif[key])
        img3 = axs[1, 0].imshow(DCTs_fake_dif[key],cmap=cm.jet,interpolation='nearest')
        axs[1, 0].set_title('Fake image',y=-0.25)
        add_text(axs[1, 0], DCTs_fake_dif[key])
        img4 = axs[1, 1].imshow(DCTs_wm_fake_dif[key],cmap=cm.jet,interpolation='nearest')
        axs[1, 1].set_title('Fake watermarked image',y=-0.25)
        add_text(axs[1, 1], DCTs_wm_fake_dif[key])
        fig.suptitle(key[:-4], fontsize=16)
        fig.savefig("Analysis_data/Annie/DCTs_comparison/"+key[:-4]+".eps")
    # plt.show()
