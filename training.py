import torch
import numpy as np
import pickle as pkl
import os
from torch.optim import Adam, RMSprop
from torch.autograd import Variable
from torch import nn
from model import StarTrekModel
from gather_data import iterate_minibatches

with open("data/train_set.pkl", "rb") as f:
    x_train, y_train = pkl.load(f)

with open("data/val_set.pkl", "rb") as f:
    x_val, y_val = pkl.load(f)

net = StarTrekModel().cuda()

criterion = nn.NLLLoss()

#torch.backends.cudnn.benchmark = True


def validation(model):
    running_loss = 0.0
    for images, labels in iterate_minibatches(x_val, y_val, batchsize=8):
        labels = Variable(torch.LongTensor(labels))
        labels = torch.argmax(labels, axis=1)
        images = torch.FloatTensor(images)
        images = images/255.
        images = images.resize(images.size(0), 3, 224, 224)
        output = model.forward(images.cuda())
        aa = output.detach().cpu().numpy()
        aa = np.argmax(np.exp(aa), axis=1)
        #print(aa)
        loss = criterion(output, labels.cuda())
        labels = labels.detach().cpu().numpy()
        #print(labels)
        accuracy = np.mean(labels == aa)
        #print(pred, labels)
        #print(output.shape, labels.shape)
        #print(labels, type(labels))
        #print(pred_labels, type(output))
        running_loss += loss.item()
        return running_loss, accuracy


def train(epochs, lr=0.01):
    optimizer = RMSprop(net.parameters(), lr=lr)
    print_every = 1
    error_l = []
    epochs_l = []
    steps = 0
    for e in range(epochs):
        running_loss = 0.0
        torch.cuda.empty_cache()
        for images, labels in iterate_minibatches(x_train, y_train, batchsize=32, shuffle=True):
            steps += 1
            labels = Variable(torch.LongTensor(labels))
            labels = torch.argmax(labels, axis=1)
            images = torch.FloatTensor(images)
            images = images/255.
            images = images.resize(images.size(0), 3, 224, 224)
            images = images.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            output = net(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if steps % print_every == 0:
                print("Epochs-- {}/{} Loss-- {} val {}".format(e+1,
                      epochs,
                      running_loss/print_every,
                      validation(net)))
                error_l.append(running_loss/print_every)
                epochs_l.append(e+1)
            running_loss = 0.0
    if os.path.exists("model/") is not True:
        os.mkdir("model/")
    elif os.path.exists("model/") is True:
        torch.save(net.state_dict(), 'model/_strek_model_save3.pt')
    return output, error_l, epochs_l


output, error_l, epochs_l = train(300, lr=0.001)
