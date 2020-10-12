import torch
import cv2
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from model import StarTrekModel
from gather_data import crop_to_face

with open("data/train_set.pkl", "rb") as f:
    x_train, y_train = pkl.load(f)

y_train = np.argmax(y_train, axis=1)

img = cv2.imread("picard.jpg")
#img = crop_to_face(img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
def Prediction(image):
    net = StarTrekModel()
    net.load_state_dict(torch.load("model/_strek_model_save3.pt"))

    net.eval()
    with torch.no_grad():
        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image/255.
        image = torch.FloatTensor(image)
        image = image.reshape(1, 3, 224, 224)
        pred = net.forward(image)
        out = torch.argmax(torch.exp(pred), dim=1)
        out = out.detach().cpu().numpy()
        print(out)
        img_index = np.where(y_train == [out[0]])[0][0]
        plt.imshow(x_train[img_index])


Prediction(img)
