import pickle as pkl
import matplotlib.pyplot as plt
import torch
from torch.optim import Adam
from model import StarTrekModel
from gather_data import iterate_minibatches


with open("data/train.pkl", "rb") as f:
    x_train, y_train = pkl.load(f)


train_loader = iterate_minibatches

epochs = 100

net = StarTrekModel().cuda()


def train(images, labels, epochs):
    optimizer = Adam(net.parameters(), lr=0.001)
    criterion = torch.nn.SmoothL1Loss()
    epochs = 100
    print_every = 100
    error_l = []
    epochs_l = []
    for e in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader(images, labels, batchsize=64)):
            img_batch, label_batch = data
            img_batch = img_batch / 255.
            img_batch = torch.FloatTensor(img_batch)
            img_batch = img_batch.reshape(img_batch.shape[0], 3, 244, 244)
            label_batch = torch.FloatTensor(label_batch)
            label_batch = label_batch.view(label_batch.size(0), -1)
            output = net(img_batch.cuda())
            loss = criterion(output, label_batch.cuda())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % print_every == 0:
                print("Epochs-- {}/{} Loss-- {}".format(e+1,
                      epochs,
                      running_loss/print_every))
                error_l.append(running_loss/print_every)
                epochs_l.append(e+1)
    return output, error_l, epochs_l

train(x_train, y_train, 100)
