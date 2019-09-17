import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import store_load as sl
import Statistics as st
import os

if __name__ == "__main__":
    # image_paths = sl.get_imagepaths_by_name(img_category)
    data_set_dct_paths = sl.get_datapaths_by_name('DCTCoeffs', 'embedder', '')
    data_set_paths = sl.get_datapaths_by_name('wm_img', 'embedder', '')
    payload = 'p_100'
    image_paths = sl.get_imagepaths_by_name('')

    f = open('Analysis_data/Sarah/Fidelity_Embedder_' + payload + '.txt', 'w')
    i = 0
    av = 0
    fidArr = list()
    for path in data_set_paths:
        # blue_channel_DCTs = sl.get_blue_channel(path)
        # av += np.average(blue_channel_DCTs)

        # Image Fidelity
        img = np.array(Image.open(path))
        img_org = np.array(Image.open(image_paths[i]))
        sum = 0
        mse = 0
        # MSE
        for dim in range(0, img.shape[2]):
            B = img[:, :, dim].astype('float32')
            A = img_org[:, :, dim].astype('float32')
            sum += (A - B) ** 2
        mse = np.sum(sum) / (3 * img.shape[0] * img.shape[1])
        # Fidelity
        fid = 10 * np.log10((255 ** 2 / mse))
        fidArr.append(fid)
        img_name = os.path.basename(path)
        img_name = img_name.split('/')[0]
        img_name = img_name.split('.')[0]
        f.write("{0}: {1}\n".format(img_name, fid))
        i += 1
    f.write("{0}: {1}".format('Average', np.average(fidArr)))
    f.close()

    data_set_paths_attacker_s = sl.get_datapaths_by_name('fake_wm_img', 'attacker', '')
    data_set_paths_attacker_c = sl.get_datapaths_by_name('fake_cw_img', 'attacker', '')
    f = open('Analysis_data/Sarah/Fidelity_Attacker_' + payload + '.txt', 'w')
    i = 0
    fidArr = list()
    for path in data_set_paths_attacker_s:
        img = np.array(Image.open(path))
        img_org = np.array(Image.open(data_set_paths_attacker_c[i]))
        sum = 0
        mse = 0
        # MSE
        for dim in range(0, img.shape[2]):
            B = img[:, :, dim].astype('float32')
            A = img_org[:, :, dim].astype('float32')
            sum += (A - B) ** 2
        mse = np.sum(sum) / (3 * img.shape[0] * img.shape[1])
        # Fidelity
        fid = 10 * np.log10((255 ** 2 / mse))
        if fid == np.inf:
            fid = 0
        fidArr.append(fid)
        img_name = os.path.basename(path)
        img_name = img_name.split('/')[0]
        img_name = img_name.split('.')[0]
        f.write("{0}: {1}\n".format(img_name, fid))
        i += 1

    f.write("{0}: {1}".format('Average', np.average(fidArr)))
    f.close()

    f = open('Analysis_data/Sarah/Chi_test_Embedder_' + payload + '.txt', 'w')
    i = 0
    chiArrCorrA = list()
    chiArrUnCorrA = list()
    chiArrCorrB = list()
    chiArrUnCorrB = list()
    for path in data_set_paths:
        img = np.array(Image.open(path))
        img_org = np.array(Image.open(image_paths[i]))
        B = img[:,:,2].astype('int32')
        A = img_org[:,:,2].astype('int32')
        a_min = A.min()
        b_min = B.min()
        a_max = A.max()
        b_max = B.max()
        a_hist = st.subhistcount(A, a_min, a_max)
        b_hist = st.subhistcount(B, b_min, b_max)
        a_chi_corr = st.chisquaretestcorr(a_hist)
        a_chi_uncorr = st.chisquaretestuncorr(a_hist)
        b_chi_corr = st.chisquaretestcorr(b_hist)
        b_chi_uncorr = st.chisquaretestuncorr(b_hist)
        chiArrCorrA.append(a_chi_corr)
        chiArrUnCorrA.append(a_chi_uncorr)
        chiArrCorrB.append(a_chi_corr)
        chiArrUnCorrB.append(b_chi_uncorr)
        img_name = os.path.basename(path)
        img_name = img_name.split('/')[0]
        img_name = img_name.split('.')[0]
        f.write("{0}:\n".format(img_name))
        f.write("{0}: {1}\n".format('Org corr', a_chi_corr))
        f.write("{0}: {1}\n".format('org uncorr', a_chi_uncorr))
        f.write("{0}: {1}\n".format('WM corr', b_chi_corr))
        f.write("{0}: {1}\n".format('WM uncorr', b_chi_uncorr))
        i += 1
    f.write("{0}: {1}\n".format('Average Org corr', np.average(chiArrCorrA)))
    f.write("{0}: {1}\n".format('Average Org uncorr', np.average(chiArrUnCorrA)))
    f.write("{0}: {1}\n".format('Average WM corr', np.average(chiArrCorrB)))
    f.write("{0}: {1}\n".format('Average WM uncorr', np.average(chiArrUnCorrB)))
    f.close()

    f = open('Analysis_data/Sarah/Chi_test_Attacker_' + payload + '.txt', 'w')
    data_set_paths_attacker_s = sl.get_datapaths_by_name('fake_wm_img', 'attacker', '')
    data_set_paths_attacker_c = sl.get_datapaths_by_name('fake_cw_img', 'attacker', '')
    i = 0
    AttchiArrCorrA = list()
    AttchiArrUnCorrA = list()
    AttchiArrCorrB = list()
    AttchiArrUnCorrB = list()
    for path in data_set_paths_attacker_s:
        img = np.array(Image.open(path))
        img_org = np.array(Image.open(data_set_paths_attacker_c[i]))
        B = img[:,:,2].astype('int32')
        A = img_org[:,:,2].astype('int32')
        a_min = A.min()
        b_min = B.min()
        a_max = A.max()
        b_max = B.max()
        a_hist = st.subhistcount(A, a_min, a_max)
        b_hist = st.subhistcount(B, b_min, b_max)
        a_chi_corr = st.chisquaretestcorr(a_hist)
        a_chi_uncorr = st.chisquaretestuncorr(a_hist)
        b_chi_corr = st.chisquaretestcorr(b_hist)
        b_chi_uncorr = st.chisquaretestuncorr(b_hist)
        AttchiArrCorrA.append(a_chi_corr)
        AttchiArrUnCorrA.append(a_chi_uncorr)
        AttchiArrCorrB.append(a_chi_corr)
        AttchiArrUnCorrB.append(b_chi_uncorr)
        img_name = os.path.basename(path)
        img_name = img_name.split('/')[0]
        img_name = img_name.split('.')[0]
        f.write("{0}:\n".format(img_name))
        f.write("{0}: {1}\n".format('Org corr', a_chi_corr))
        f.write("{0}: {1}\n".format('org uncorr', a_chi_uncorr))
        f.write("{0}: {1}\n".format('WM corr', b_chi_corr))
        f.write("{0}: {1}\n".format('WM uncorr', b_chi_uncorr))
        i += 1
    f.write("{0}: {1}\n".format('Average Org corr', np.average(AttchiArrCorrA)))
    f.write("{0}: {1}\n".format('Average Org uncorr', np.average(AttchiArrUnCorrA)))
    f.write("{0}: {1}\n".format('Average WM corr', np.average(AttchiArrCorrB)))
    f.write("{0}: {1}\n".format('Average WM uncorr', np.average(AttchiArrUnCorrB)))

    f.close()

    # no of changed pixels
    # org vs wm
    # fake org vs fake wm
    f = open('Analysis_data/Sarah/Changed_pixels_' + payload + '.txt', 'w')
    i = 0
    f.write("Original vs. WMed\n")
    for path in data_set_paths:
        img = np.array(Image.open(path))
        img_org = np.array(Image.open(image_paths[i]))
        sum = 0
        for dim in range(img.shape[2]):
            A = img_org[:,:,dim].astype('float32')
            B = img[:,:,dim].astype('float32')
            C = abs(B - A)
            sum += np.sum(C[C != 0])
        i += 1
        img_name = os.path.basename(path)
        img_name = img_name.split('/')[0]
        img_name = img_name.split('.')[0]
        f.write("{0}: {1}\n".format(img_name, sum))

    data_set_paths_attacker_s = sl.get_datapaths_by_name('fake_wm_img', 'attacker', '')
    data_set_paths_attacker_c = sl.get_datapaths_by_name('fake_cw_img', 'attacker', '')
    f.write("Fake original vs. fake WMed\n")
    i = 0
    for path in data_set_paths_attacker_s:
        img = np.array(Image.open(path))
        img_org = np.array(Image.open(data_set_paths_attacker_c[i]))
        sum = 0
        for dim in range(img.shape[2]):
            A = img_org[:,:,dim].astype('float32')
            B = img[:,:,dim].astype('float32')
            C = abs(B - A)
            sum += np.sum(C[C != 0])
        i += 1
        img_name = os.path.basename(path)
        img_name = img_name.split('/')[0]
        img_name = img_name.split('.')[0]
        f.write("{0}: {1}\n".format(img_name, sum))

    f.write("Original vs fake original\n")
    i = 0
    for path in data_set_paths:
        img_org = np.array(Image.open(path))
        img = np.array(Image.open(data_set_paths_attacker_c[i]))
        sum = 0
        for dim in range(img.shape[2]):
            A = img_org[:,:,dim].astype('float32')
            B = img[:,:,dim].astype('float32')
            C = abs(B - A)
            sum += np.sum(C[C != 0])
        i += 1
        img_name = os.path.basename(path)
        img_name = img_name.split('/')[0]
        img_name = img_name.split('.')[0]
        f.write("{0}: {1}\n".format(img_name, sum))
    f.close()
    
    # no of successful attacks & similarity values
    data_set_paths_st_fake = sl.get_datapaths_by_name('fake_1', 'detector', '')
    data_set_paths_st = sl.get_datapaths_by_name('orig_1', 'detector', '')
    f = open('Analysis_data/Sarah/Detection_results_' + payload + '.txt', 'w')
    embed = list()
    attack = list()
    i= 0
    for path in data_set_paths_st:
        x = np.load(path)
        y = np.load(data_set_paths_st_fake[i])
        embed.append(x[0])
        attack.append(y[0])
        img_name = os.path.basename(path)
        img_name = img_name.split('/')[0]
        img_name = img_name.split('.')[0]
        f.write("{0}:\n".format(img_name))
        f.write("{0}: {1}\n".format('Original', x[0]))
        f.write("{0}: {1}\n".format('Attack', y[0]))
        f.write("{0}: {1}\n".format('Original', x[1]))
        f.write("{0}: {1}\n".format('Fake', y[1]))
        i += 1
    embed = np.asarray(embed)
    attack = np.asarray(attack)
    f.write("{0}: {1}\n".format('Original min', embed.min()))
    f.write("{0}: {1}\n".format('Original max', embed.max()))
    f.write("{0}: {1}\n".format('Original average', np.average(embed)))
    f.write("{0}: {1}\n".format('Attack min', attack.min()))
    f.write("{0}: {1}\n".format('Attack max', attack.max()))
    f.write("{0}: {1}\n".format('Average attack', np.average(attack)))

    f.close()

    # irgendwas mit den watermarks
    data_set_paths_wm = sl.get_datapaths_by_name('wm', 'embedder', '')
    path_wm = sl.get_datapaths_by_name('fake_wm', 'attacker', '')
    wmArr = list()
    wmFakeArr = list()
    i = 0
    diff = 0
    f = open('Analysis_data/Sarah/Watermarks_' + payload + '.txt', 'w')
    for path in data_set_paths_wm:
        wmArr.append(np.load(path))
        wmFakeArr.append(np.load(path_wm[i]))
    npArr = np.asarray(wmArr)
    npArrF = np.asarray(wmFakeArr)
    f.write("{0}: {1}\n".format("Org min", npArr.min()))
    f.write("{0}: {1}\n".format("Fake min", npArr.max()))
    f.write("{0}: {1}\n".format("Org max", npArrF.min()))
    f.write("{0}: {1}\n".format("Fake max", npArrF.max()))
    f.write("{0}: {1}\n".format("Org avg", np.average(npArr)))
    f.write("{0}: {1}\n".format("Fake avg", np.average(npArrF)))
    f.close()
