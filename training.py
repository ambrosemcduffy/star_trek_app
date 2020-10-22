import torch
import numpy as np
import pickle as pkl
import os
import matplotlib.pyplot as plt
from torch.optim import Adam, RMSprop, Adamax, Adagrad, SGD
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
print(x_train.shape[0])


def validation(model):
    running_loss = 0.0
    for images, labels in iterate_minibatches(x_val, y_val, batchsize=32, shuffle=True):
        labels = Variable(torch.LongTensor(labels))
        labels = torch.argmax(labels, axis=1)
        images = torch.FloatTensor(images)
        images = images/255.
        images = images.resize(images.size(0), 3, 224, 224)
        output = model.forward(images.cuda())
        loss = criterion(output, labels.cuda())
        labels = labels.detach().cpu().numpy()
        pred = output.detach().cpu().numpy()
        pred = np.argmax(np.exp(pred), axis=1)
        accuracy = np.mean(labels == pred)
        running_loss += loss.item()
        return running_loss, accuracy


def train(epochs, lr=0.01):
    optimizer = RMSprop(net.parameters(), lr=lr)
    print_every = 1
    error_l = []
    error_l_test = []
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
            test_loss, accuracy = validation(net)
            error_l_test.append(test_loss)
            if steps % print_every == 0:
                print("Epochs-- {}/{} Loss-- {} val {}".format(e+1,
                      epochs,
                      running_loss/print_every,
                      (test_loss/print_every, accuracy)))
                error_l.append(running_loss/print_every)
                epochs_l.append(e+1)
            running_loss = 0.0
    if os.path.exists("model/") is not True:
        os.mkdir("model/")
    elif os.path.exists("model/") is True:
        torch.save(net.state_dict(), 'model/_strek_model_save4.pt')
    return output, error_l, epochs_l, error_l_test


output, error_l, epochs_l, error_l_test = train(150, lr=0.007)

def plot_loss(epochs_l, error, error_test):
    plt.plot(epochs_l, error)
    plt.plot(epochs_l, error_l_test)
    plt.show()


plot_loss(epochs_l, error_l, error_l_test)
