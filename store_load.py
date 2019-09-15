"""
Set of functions to store and load intermediate data values.
"""

import os
import numpy as np

from PIL import Image


def get_imagepaths_by_name(name):
    """
    Returns a list of paths of images having 'name' in their name.

    Searchs in 'images/Originals/'.

    name: string

    Returns a list of paths of found filenames.
    """

    images = os.listdir('images/Originals/')
    filtered = [img for img in images if name in img]
    res = []
    for img in filtered:
        res.append(os.path.join('images/Originals', img))
    return res

def get_datapaths_by_name(data_type, role, name):
    """
    Returns a list of paths of files having 'name' in their name.

    Searchs in 'Analysis_data/role/data_type/'.

    data_type: string, e.g.: "DCT" or "wm"
    role: string, used to identify a subfolder, e.g. embedder, detector,
          attacker
    name: string

    Returns a list of paths of found filenames.
    """

    base_path = 'Analysis_data/' + role + '/' + data_type + '/'
    files = os.listdir(base_path)
    filtered = [file for file in files if name in file]
    res = []
    for file in filtered:
        res.append(os.path.join(base_path, file))
    return res

def save_data(data, path_to_file, role, data_type):
    """
    Saves data in a file and overwrites the file if it already exists.
    
    Files are stored in: 'Analysis_data/data_type/file_name.npy'.

    data: numpy.array, list, string
    path_to_file: string, used to extract the filename to name the file
    role: string, used to create a subfolder, e.g. embedder, detector, attacker
    data_type: string, type to identify the kind of data stored, e.g. "DCT"
                or "wm"
    """
    base_path = 'Analysis_data/' + role + '/' + data_type + '/'
    # get filename
    image_name = os.path.basename(path_to_file)
    # remove endings like '.jpg'
    image_name = image_name.split('.')[0]
    # create folder structure if not already existing
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    np.save(base_path + image_name, data)

def save_nparray_as_img(data, path_to_file, role, data_type):
    """
    Saves data as an image and overwrites the image if it already exists.
    
    Files are stored in: 'Analysis_data/data_type/file_name.jpg'.

    data: numpy.array
    path_to_file: string, used to extract the filename to name the image
    role: string, used to create a subfolder, e.g. embedder, detector, attacker
    data_type: string, type to identify the kind of data stored, e.g. "DCT"
                or "wm"
    """
    base_path = 'Analysis_data/' + role + '/' + data_type + '/'
    # get filename
    image_name = os.path.basename(path_to_file)
    # remove endings like '.jpg'
    image_name = image_name.split('.')[0]
    # create folder structure if not already existing
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    wm_img = Image.fromarray(data.astype('uint8'), mode='RGB')
    wm_img.save(base_path + image_name + '.jpg')

# added this function for completeness but it is actually not very useful
# because it only wraps np.load()
def load_data(path_to_file):
    """
    Loads data from a file and returns it.
    """
    return np.load(path_to_file)

if __name__=="__main__":
    bw_paths = get_imagepaths_by_name('black_white_')
    role = 'embedder'
    for img in bw_paths:
        l = 100
        w = np.random.normal(0.0, 1.0, (1, l))
        save_data(w, img, role, 'wm')
    bw_wm = get_datapaths_by_name('wm', role, 'black_white_')
    for wm in bw_wm:
        print(load_data(wm))
