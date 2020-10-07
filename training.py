import pickle as pkl
import numpy as np
import torch
from torchvision import transforms, datasets
import os
from torch.optim import Adam, RMSprop, SGD
from torch.autograd import Variable
from torch import nn
from model import StarTrekModel
from gather_data import iterate_minibatches as data_loader

with open("data/train.pkl", "rb") as f:
    x_train, y_train = pkl.load(f)

y_train = np.argmax(y_train, axis=1)
net = StarTrekModel().cuda()

criterion = nn.CrossEntropyLoss()


def train(epochs, lr=0.01):
    optimizer = RMSprop(net.parameters(), lr=lr)
    print_every = 1
    error_l = []
    epochs_l = []
    steps = 0
    for e in range(epochs):
        running_loss = 0.0
        torch.cuda.empty_cache()
        for images, labels in data_loader(x_train, y_train, 32, shuffle=True):
            steps += 1
            labels = Variable(torch.LongTensor(labels))
            images = torch.FloatTensor(images) / 255.
            images = images.cuda()
            labels = labels.cuda()
            images = images.resize(32, 3, 224, 224)
            optimizer.zero_grad()
            output = net(images)
            loss = criterion(output, labels)
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


output, error_l, epochs_l = train(300, lr=0.00001)
