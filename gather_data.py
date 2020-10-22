import os
import glob
import shutil
import pickle as pkl
from collections import defaultdict

import numpy as np
import pandas as pd
import cv2
import PIL
import matplotlib.pyplot as plt

from google_images_download import google_images_download

# Importing in the Cascade file for face Detection.
cas_path = "data/haarcascade_frontalface_default.xml"


def download_images(limit=20, download=False):
    print("downloading {} images for each".format(limit))
    """ This function downloads images in based on keywords provided,
    from a text file in the downloads dir.
    The parameters are:
        n_names: can provide number of to input
        limit: can provided the number of images to download per name
        download: if True it will begin to download from google.
    Returns: None
    """
    keyword_l = []
    # Reading in the names.txt file, and append names to a list.
    with open("data/names.txt", "r+") as f:
        names = f.readlines()
        for name in names:
            keyword_l.append(name.rstrip("\n"))

    n_names = len(keyword_l)
    cnt = 0
    if download is True:
        # If downloads folder exist remove it.
        if os.path.exists("downloads/") is True:
            shutil.rmtree("downloads/")
        # Download images.
        response = google_images_download.googleimagesdownload()
        for k in keyword_l:
            if cnt == n_names:
                break
            arguments = {"keywords": k+" Star Trek",
                         "limit": limit,
                         "print_urls": True}
            response.download(arguments)
            cnt += 1
    return None


def get_data(file_dir, size=224):
    """ This function returns a dictionary containing,
    data : images array for each character.
    data_int: labels array for each character
    """

    # Creating dictionary to store image files
    data = defaultdict(list)
    # Obtaining the names of star trek character, and import images
    file_dir = glob.glob("{}*".format(file_dir))
    for folder in file_dir:
        name = folder.split("\\")[1].split("Star Trek")[0].rstrip()
        for file in glob.glob(folder+"/*"):
            img = PIL.Image.open(file)
            img = np.array(img)
            if len(img.shape) == 3:
                if img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                # Cropping the image to square dims
                img = cv2.resize(img, (size, size))
                img_arr = np.array(img)
                data[name].append(np.array(img_arr))
        # obtaining target dummy variables
    data_int = defaultdict(list)
    for k, v in data.items():
        for i in range(len(v)):
            dummies = pd.get_dummies(list(data.keys()))[k].values
            data_int[k].append(dummies)
        data_int[k] = np.array(data_int[k])
    return data, data_int


def create_dataset(data, data_int, name):
    """ This function constructs a dataset from dictionaries
    then saves the trainset and labels out as a pkl file.
    Returns:
        x_train: image array
        y_train: labels array
    """
    x = []
    y = []
    for k, v in data.items():
        x.append(np.array(data[k]))
        y.append(np.array(data_int[k]))
    x_train = np.vstack(x)
    y_train = np.vstack(y)
    with open("data/{}.pkl".format(name), "wb") as f:
        pkl.dump([x_train, y_train], f)
    return x_train, y_train


def display_images(images, targets):
    """ This function take in an array of images, and
    displays for exploratory Analysis.
    Return:
        None
    """
    fig = plt.figure(figsize=(8, 8))
    columns = 4
    rows = 5
    #targets = np.argmax(targets, axis=1)
    for i in range(1, columns*rows+1):
        num = np.random.randint(images.shape[0])
        fig.add_subplot(rows, columns, i)

        if type(images) != "torch.Tensor":
            plt.imshow(images[num], cmap="gray")
            plt.xticks([])
            plt.yticks([])
            plt.title(targets[num])
        else:
            plt.imshow(images[num])
            plt.xticks([])
            plt.yticks([])
            plt.title("hey")
            plt.tight_layout()
    plt.show()
    return None


def crop_to_face(image):
    """This function uses haarCascades to detect facial features,
    after the features are detected we cropp the image into ROI.
    Return:
        roi:cropped detected face image in square dimensions
        image_copy: if roi is not detected returns original image.
    """

    # Importing the cascade file into the Classifier.
    face_cas = cv2.CascadeClassifier(cas_path)
    # Setting the parameters.
    faces = face_cas.detectMultiScale(image, 1.05, 10)
    image_with_detection = image.copy()
    image_copy = image.copy()
    # Draw a box around the face
    p = 5
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            cv2.rectangle(image_with_detection,
                          (x, y),
                          (x+w, y+h),
                          (255, 0, 0),
                          3)
            roi = image_copy[y-p: y+h+p, x-p: x+w+p]
            return roi
    else:
        return image_copy


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    """This function creates a custom DataLoader
    parameters:
        inputs: image array dataset
        targets: label array dataset
        batchsize: [64, 128, 256]
        shuffle: if True shuffles the dataset.
    yields: image array, labels array of batchsize.
    """
    assert inputs.shape[0] == targets.shape[0]
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def save_data(file_train, file_val):
    data, data_int = get_data(file_train)
    data_val, data_val_int = get_data(file_val)
    x_train, y_train = create_dataset(data, data_int, "train_set")
    x_val, y_val = create_dataset(data_val, data_val_int, "val_set")
    return (x_train, y_train), (x_val, y_val)
