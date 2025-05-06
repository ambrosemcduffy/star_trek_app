import torch
import glob
import cv2
import matplotlib.pyplot as plt
from model import vgg16_pretrain
from torchvision import transforms


# Use Metal (MPS) if available
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
    print("✅ Using Metal (MPS) backend.")
else:
    device = torch.device("cpu")
    print("⚠️ MPS not available. Using CPU.")


# Setting Normalization parameters for the pretrained network.
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

vgg16_model = vgg16_pretrain()
vgg16_model.load_state_dict(torch.load("model/_strek_model_save1.pt", map_location=torch.device("mps" if torch.backends.mps.is_available() else "cpu")))

# Importing the names of characters.
with open("data/class_names.txt", "r+") as f:
    names_l = f.readlines()


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
    """ This function predicts based on which star trek character,
    one looks like.
    Return:
        name of the character one looks like.
    """
    vgg16_model.eval()
    with torch.no_grad():
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = image.copy()
        img = cv2.resize(img, (224, 224))
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean, std)(img)
        img = img.reshape(1, 3, 224, 224)
        pred = vgg16_model(img.to(device))
        pred = torch.argmax(torch.exp(pred), axis=1)
        name = names_l[pred.detach().cpu().numpy()[0]].strip()
        image_ = cv2.imread("./data/{}.jpg".format(name))
        image_ = cv2.cvtColor(image_, cv2.COLOR_BGR2RGB)
        images = [image, image_]
        display_images(images, name)
        return name


Predict("./testRiinu.jpeg")