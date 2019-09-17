from PIL import Image, ImageFilter
import numpy as np
import os

import store_load as sl
import Detector


def evaluate_rotation(img_path, angle, role, alpha, sameSeed, orig_img):
    img = Image.open(orig_img)
    img_name = os.path.basename(img_path)
    img_name = img_name.split('.')[0]
    data_paths = sl.get_datapaths_by_name('wm', 'embedder', img_name)[0]
    w = np.load(data_paths)
    data_paths = sl.get_datapaths_by_name('rotated_by_' + str(angle), role, img_name)[0]
    s = Image.open(data_paths)
    detect_result = Detector.detect(w, s, img, alpha, sameSeed)
    # STORE DATA
    # sl.save_data(detect_result, img_path, role, 'rotated_' + str(sameSeed))
    print("test against image rotated by " + str(angle) + " degrees:" + img_name)
    print(detect_result)

    data_paths = sl.get_datapaths_by_name('wm_img', 'embedder', img_name)[0]
    s = Image.open(data_paths)
    detect_result = Detector.detect(w, s, img, alpha, sameSeed)
    # STORE DATA
    # sl.save_data(detect_result, img_paths, role, 'orig_' + str(sameSeed))
    print("test against watermarked image:" + img_name)
    print(detect_result)


def evaluate_gaussian_filter(img_path, filter_size, role, alpha, sameSeed, orig_img):
    img = Image.open(orig_img)
    img_name = os.path.basename(img_path)
    img_name = img_name.split('.')[0]
    data_paths = sl.get_datapaths_by_name('wm', 'embedder', img_name)[0]
    w = np.load(data_paths)
    data_paths = sl.get_datapaths_by_name('filtered_by_' + str(filter_size), role, img_name)[0]
    s = Image.open(data_paths)
    detect_result = Detector.detect(w, s, img, alpha, sameSeed)
    # STORE DATA
    # sl.save_data(detect_result, img_path, role, 'rotated_' + str(sameSeed))
    print("test against image filtered by size " + str(filter_size) + ":" + img_name)
    print(detect_result)

    data_paths = sl.get_datapaths_by_name('wm_img', 'embedder', img_name)[0]
    s = Image.open(data_paths)
    detect_result = Detector.detect(w, s, img, alpha, sameSeed)
    # STORE DATA
    # sl.save_data(detect_result, img_paths, role, 'orig_' + str(sameSeed))
    print("test against watermarked image:" + img_name)
    print(detect_result)


def evaluate_sharpen(img_path, role, alpha, sameSeed, orig_img):
    img = Image.open(orig_img)
    img_name = os.path.basename(img_path)
    img_name = img_name.split('.')[0]
    data_paths = sl.get_datapaths_by_name('wm', 'embedder', img_name)[0]
    w = np.load(data_paths)
    data_paths = sl.get_datapaths_by_name('sharpen', role, img_name)[0]
    s = Image.open(data_paths)
    detect_result = Detector.detect(w, s, img, alpha, sameSeed)
    # STORE DATA
    # sl.save_data(detect_result, img_path, role, 'rotated_' + str(sameSeed))
    print("test against image sharpening:" + img_name)
    print(detect_result)

    data_paths = sl.get_datapaths_by_name('wm_img', 'embedder', img_name)[0]
    s = Image.open(data_paths)
    detect_result = Detector.detect(w, s, img, alpha, sameSeed)
    # STORE DATA
    # sl.save_data(detect_result, img_paths, role, 'orig_' + str(sameSeed))
    print("test against watermarked image:" + img_name)
    print(detect_result)


def test_against_rotation(img_path, angle, sameSeed, alpha, role):
    print("robustness test against rotation:")
    img = Image.open(img_path)
    rotated_img = img.rotate(angle)
    sl.save_nparray_as_img(np.array(rotated_img), img_path, role, 'rotated_by_' + str(angle))
    # rotated_img.show()
    # extract name of image to investigate from current path
    img_name = os.path.basename(img_path)
    # remove endings
    img_name = img_name.split('.')[0]
    orig_img = sl.get_imagepaths_by_name(img_name)[0]
    evaluate_rotation(img_path, angle, role, alpha, sameSeed, orig_img)


def test_against_gaussian_filter(img_path, filter_size, sameSeed, alpha, role):
    print("robustness test against gaussian filter:")
    img = Image.open(img_path)
    filtered_img = img.filter(ImageFilter.GaussianBlur(radius=filter_size))
    sl.save_nparray_as_img(np.array(filtered_img), img_path, role, 'filtered_by_' + str(filter_size))
    # rotated_img.show()
    # extract name of image to investigate from current path
    img_name = os.path.basename(img_path)
    # remove endings
    img_name = img_name.split('.')[0]
    orig_img = sl.get_imagepaths_by_name(img_name)[0]
    evaluate_gaussian_filter(img_path, filter_size, role, alpha, sameSeed, orig_img)


def test_against_sharpen(img_path, sameSeed, alpha, role):
    print("robustness test against sharpening:")
    img = Image.open(img_path)
    """
    From https://pythontic.com/image-processing/pillow/sharpen-filter
    
    The convolution matrix used is,
        (-2, -2, -2,
        -2, 32, -2,
        -2, -2, -2)
    
    a 3x3 matrix.
    """
    filtered_img = img.filter(ImageFilter.SHARPEN)
    sl.save_nparray_as_img(np.array(filtered_img), img_path, role, 'sharpen')
    # rotated_img.show()
    # extract name of image to investigate from current path
    img_name = os.path.basename(img_path)
    # remove endings
    img_name = img_name.split('.')[0]
    orig_img = sl.get_imagepaths_by_name(img_name)[0]
    evaluate_sharpen(img_path, role, alpha, sameSeed, orig_img)


def run_robustness_tests():
    sameSeed = 1
    alpha = 0.04
    role = 'robustness'
    img_category = 'low_contrast_'
    rotation_angle = 0
    filter_size = 5

    print("robustness tests")
    image_paths = sl.get_datapaths_by_name('wm_img', 'embedder', img_category)
    for img_path in image_paths:
        # test_against_rotation(img_path, rotation_angle, sameSeed, alpha, role)
        # test_against_gaussian_filter(img_path, filter_size, sameSeed, alpha, role)
        test_against_sharpen(img_path, sameSeed, alpha, role)


if __name__ == "__main__":
    run_robustness_tests()
