import torch
import cv2
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from model import StarTrekModel
from gather_data import crop_to_face, display_images

with open("data/val_set.pkl", "rb") as f:
    x_train, y_train = pkl.load(f)

y_train = np.argmax(y_train, axis=1)


def Prediction(img):
    preds = []
    net = StarTrekModel()
    net.load_state_dict(torch.load('model/_strek_model_save4.pt'))
    net.eval()
    with torch.no_grad():
        if len(img.shape) < 4:
            image = cv2.resize(img, (224, 224))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.expand_dims(image, axis=0)
            image = image.reshape(image.shape[0], 3, 224, 224)
            image = image/255.
            image = torch.FloatTensor(image)
            pred = net.forward(image)
            out = torch.argmax(torch.exp(pred), dim=1)
            out = out.detach().cpu().numpy()
            return out
        elif len(img.shape) == 4:
            for i in range(img.shape[0]):
                image = img[i]
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                #image = crop_to_face(image)
                if image.shape[:2] != (224, 224):
                    image = cv2.resize(image, (224, 224))
                    print(image.shape)
                image = np.expand_dims(image, axis=0)
                image = image.reshape(image.shape[0],
                                      3,
                                      224,
                                      224)
                image = image/255.
                image = torch.FloatTensor(image)
                pred = net.forward(image)
                out = torch.argmax(torch.exp(pred), dim=1)
                out = out.detach().cpu().numpy()
                preds.append(out[0])
            return preds


out = Prediction(x_train)
print(str((out == y_train).mean() * 100)[:4] + " %")
display_images(x_train, out)


