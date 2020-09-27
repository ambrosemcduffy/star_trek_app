import os
import glob
import shutil
import pickle as pkl

from PIL import Image
import numpy as np
import pandas as pd
from collections import defaultdict

from google_images_download import google_images_download

keyword_l = []
with open("data/names.txt", "r+") as f:
    names = f.readlines()
    for name in names:
        keyword_l.append(name.rstrip("\n"))


def download_images(n_images=5, limit=20, download=False):
    """
    Parameters
    ----------
    n_images : TYPE, optional
        DESCRIPTION. The default is 5.
    limit : TYPE, optional
        DESCRIPTION. The default is 20.
    download : TYPE, optional
        DESCRIPTION. The default is False.
    Returns
    -------
    None
    """
    cnt = 0
    if download is True:
        if os.path.exists("downloads/") is True:
            shutil.rmtree("downloads/")
        response = google_images_download.googleimagesdownload()
        for k in keyword_l:
            if cnt == n_images:
                break
            arguments = {"keywords": k+" Star Trek",
                         "limit": limit,
                         "print_urls": True}
            response.download(arguments)
            cnt += 1
    return None


def get_data():
    """
    Returns
    -------
    data : Dictionary
        This function returns data dict with image arrays
    data_int : Dictionary
        this function returns data_int that contains target arrays
    """
    # creating dictionary to store image files
    data = defaultdict(list)
    # obtaining the names of star trek character, and import images
    file_dir = glob.glob("downloads/*")
    for folder in file_dir:
        name = folder.split("/")[1].split("Star Trek")[0].rstrip()
        for file in glob.glob(folder+"/*"):
            img = Image.open(file)
            img = img.resize((244, 244))
            img_arr = np.array(img)
            if img_arr.shape == (244, 244, 3):
                data[name].append(np.array(img_arr))
    # obtaining target dummy variables
    data_int = defaultdict(list)
    for k, v in data.items():
        for i in range(len(v)):
            dummies = pd.get_dummies(list(data.keys()))[k].values
            data_int[k].append(dummies)
        data_int[k] = np.array(data_int[k])
    return data, data_int


def export_dataset():
    data, data_int = get_data()
    x = []
    y = []
    for k, v in data.items():
        x.append(np.array(data[k]))
        y.append(np.array(data_int[k]))
    x_train = np.vstack(x)
    y_train = np.vstack(y)
    with open("data/train.pkl", "wb") as f:
        pkl.dump([x_train, y_train], f)


export_dataset()
