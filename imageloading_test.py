import torch
import numpy as np
import glob
import torchvision
from gather_data import display_images
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

glob.glob("downloads/*/*.jpg")
transform = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(244),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomVerticalFlip(),
                                transforms.ToTensor()])
star_trek_data = datasets.ImageFolder("downloads/", transform=transform)
data_loader = torch.utils.data.DataLoader(star_trek_data,
                                          batch_size=256,
                                          shuffle=True)

dataiter = iter(data_loader)
x, y = dataiter.next()
x_train = [np.array(transforms.ToPILImage()(img)) for img in x]

display_images(np.array(x_train))