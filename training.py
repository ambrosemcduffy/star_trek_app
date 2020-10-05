import pickle as pkl
import numpy as np
import torch
from torchvision import transforms, datasets
import os
from torch.optim import Adam, RMSprop, SGD
from torch import nn
from model import StarTrekModel
from gather_data import iterate_minibatches as data_loader

transform = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(244),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomVerticalFlip(),
                                transforms.ToTensor()])
star_trek_data = datasets.ImageFolder("downloads/", transform=transform)
data_loader = torch.utils.data.DataLoader(star_trek_data,
                                          batch_size=64,
                                          shuffle=False)
net = StarTrekModel().cuda()

criterion = nn.CrossEntropyLoss()


def train(epochs, lr=0.01):
    optimizer = Adam(net.parameters(), lr=lr)
    print_every = 1
    error_l = []
    epochs_l = []
    steps = 0
    for e in range(epochs):
        running_loss = 0.0
        torch.cuda.empty_cache()
        for images, labels in data_loader:
            steps += 1
            images = images / 255.
            optimizer.zero_grad()
            output = net(images.cuda())
            loss = criterion(output, labels.cuda())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if steps % print_every == 0:
                print("Epochs-- {}/{} Loss-- {}".format(e+1,
                      epochs,
                      running_loss/print_every))
                error_l.append(running_loss/print_every)
                epochs_l.append(e+1)
            running_loss = 0.0
    if os.path.exists("model/") is not True:
        os.mkdir("model/")
    elif os.path.exists("model/") is True:
        torch.save(net.state_dict(), 'model/_strek_model_save.pt')
    return output, error_l, epochs_l


output, error_l, epochs_l = train(70, lr=0.0001)
