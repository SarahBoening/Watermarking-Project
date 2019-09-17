from PIL import Image
import store_load as sl
import numpy as np
import os
import Detector


def evaluate_rotation(img_path, angle, role, alpha, sameSeed):
    img = Image.open(img_path)
    # extract name of image to investigate from current path
    img_name = os.path.basename(img_path)
    # remove endings
    img_name = img_name.split('.')[0]
    # LOAD DATA
    # we need to select the first item because it is still a list eventhough it has
    # only one item
    data_paths = sl.get_datapaths_by_name('wm', 'embedder', img_name)[0]
    w = np.load(data_paths)
    # we need to select the first item because it is still a list eventhough it has
    # only one item
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


def test_against_rotation(img_path, angle, sameSeed, alpha, role):
    print("robustness test against rotation:")
    img = Image.open(img_path)
    rotated_img = img.rotate(angle)
    sl.save_nparray_as_img(np.array(rotated_img), img_path, role, 'rotated_by_' + str(angle))
    # rotated_img.show()
    evaluate_rotation(img_path, angle, role, alpha, sameSeed)


def run_robustness_tests():
    sameSeed = 1
    alpha = 0.04
    role = 'robustness'
    img_category = 'high_contrast_'
    rotation_angle = 5

    print("robustness tests")
    image_paths = sl.get_datapaths_by_name('wm_img', 'embedder', img_category)
    for img_path in image_paths:
        test_against_rotation(img_path, rotation_angle, sameSeed, alpha, role)


if __name__ == "__main__":
    run_robustness_tests()
