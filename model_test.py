import torch
import cv2
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from model import StarTrekModel

cas_path = "data/haarcascade_frontalface_default.xml"


with open("data/train.pkl", "rb") as f:
    x_train, y_train = pkl.load(f)
y_train = np.argmax(y_train, axis=1)


img1 = cv2.imread("jim2.jpg")
img2 = cv2.imread("jim.jpg")
img3 = cv2.imread("kirk.jpg")
img4 = cv2.imread("spock.jpg")

img6 = cv2.imread("spock2.jpg")
img5 = cv2.imread("vulcan.jpg")

img7 = cv2.imread("spock_you.jpg")
img8 = cv2.imread("spock_hal.jpg")


def Prediction(image):
    net = StarTrekModel()
    net.load_state_dict(torch.load("model/_strek_model_save.pt"))

    net.eval()

    crop_img = cv2.resize(image, (224, 224))
    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    plt.imshow(crop_img)
    crop_img = crop_img/255.
    crop_img = torch.FloatTensor(crop_img)
    crop_img = crop_img.reshape(1, 3, 224, 224)
    pred = net.forward(crop_img)

    out = pred.detach().cpu().numpy()
    class_labels = np.argmax(out, axis=1)
    print(class_labels)
    if class_labels == [0]:
        print("You are Kirk")
    elif class_labels == [1]:
        print("You are Spock")

    img_index = np.where(y_train == [class_labels[0]])[0][0]
    plt.imshow(x_train[img_index])


Prediction(img8)