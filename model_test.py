import torch
import cv2
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from model import StarTrekModel
from gather_data import crop_to_face

cas_path = "data/haarcascade_frontalface_default.xml"


with open("data/train.pkl", "rb") as f:
    x_train, y_train = pkl.load(f)
y_train = np.argmax(y_train, axis=1)

net = StarTrekModel()
net.load_state_dict(torch.load("model/_strek_model_save.pt"))

net.eval()
img = cv2.imread("old_spock.jpg")
crop_img = cv2.resize(img, (224, 224))
crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
plt.imshow(crop_img)
crop_img = crop_img/255.
crop_img = torch.FloatTensor(crop_img)
crop_img = crop_img.reshape(1, 3, 224, 224)
pred = net.forward(crop_img)

out = pred.detach().cpu().numpy()
class_labels = np.argmax(out, axis=1)
print(class_labels)

img_index = np.where(y_train==[class_labels[0]])[0][0]
plt.imshow(x_train[img_index])

#test_i = int(x_train.shape[0] * .2)
#np.random.choice(idx, test_i)
