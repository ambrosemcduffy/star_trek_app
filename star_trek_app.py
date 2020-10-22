import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from model import vgg16_pretrain
from gather_data import crop_to_face
from torchvision import transforms


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

vgg16_model = vgg16_pretrain()
vgg16_model.load_state_dict(torch.load("model/_strek_model_save1.pt"))

with open("data/class_names.txt", "r+") as f:
    l = f.readlines()


def display_images(images, name):
    """ This function take in an array of images, and
    displays for exploratory Analysis.
    Return:
        None
    """
    fig = plt.figure(figsize=(8, 8))
    columns = 2
    rows = 1
    for i in range(1, columns*rows+1):
        fig.add_subplot(rows, columns, i)
        fig.suptitle(" Your Star Trek Look alike is {}".format(name), y=0.1)
        plt.imshow(images[i-1], cmap="gray")
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.show()
    return None


def Predict(image):
    vgg16_model.eval()
    with torch.no_grad():
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = image.copy()
        img = cv2.resize(img, (224, 224))
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean, std)(img)
        img = img.reshape(1, 3, 224, 224)
        pred = vgg16_model(img.cuda())
        pred = torch.argmax(torch.exp(pred), axis=1)
        name = l[pred.detach().cpu().numpy()[0]].strip()
        image_ = cv2.imread("data/{}.jpg".format(name))
        image_ = cv2.cvtColor(image_, cv2.COLOR_BGR2RGB)
        images = [image, image_]
        display_images(images, name)
        return name
