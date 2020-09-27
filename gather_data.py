import os
import glob
import shutil

from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import defaultdict

from google_images_download import google_images_download

keyword_l = []
with open("data/names.txt", "r+") as f:
    names = f.readlines()
    for name in names:
        keyword_l.append(name.rstrip("\n"))


def download_images(n_images=5, limit=20, download=False):
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
    d = defaultdict(list)
    file_dir = glob.glob("downloads/*")
    for folder in file_dir:
        name = folder.split("/")[1].split("Star Trek")[0].rstrip()
        for file in glob.glob(folder+"/*"):
            img = Image.open(file)
            d[name].append(np.array([np.array(img), name]))
    return d


#download_images(download=False)
data = get_data()
