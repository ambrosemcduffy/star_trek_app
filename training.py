import torch
from torch import nn, optim
from collections import OrderedDict
from torchvision import transforms, datasets, models
from gather_data import display_images
import numpy as np
import cv2
import matplotlib.pyplot as plt

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


transform = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=mean, std=std)])


train_data = datasets.ImageFolder("train_set/", transform=transform)
train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=32,
                                           shuffle=True)

test_data = datasets.ImageFolder("validation_set/", transform=transform)
test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=32,
                                          shuffle=True)

x_train, y_train = iter(train_loader).next()
model = models.vgg16(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

classifer_dict = OrderedDict([("fc1", nn.Linear(25088, 128)),
                              ("Relu", nn.ReLU(inplace=True)),
                              ("Dropout1", nn.Dropout(0.5)),
                              ("fc2", nn.Linear(128, 256)),
                              ("Relu", nn.ReLU(inplace=True)),
                              ("Dropout2", nn.Dropout(0.25)),
                              ("f3", nn.Linear(256, 17)),
                              ("Softmax", nn.LogSoftmax(dim=1))])
classifier = nn.Sequential(classifer_dict)

model.classifier = classifier.cuda()
model = model.cuda()

optimizer = optim.Adam(model.parameters(), lr=0.0007)
criterion = nn.NLLLoss()


def validation(test_loader, model):
    running_loss = 0.0
    for images, labels in test_loader:
        output = model.forward(images.cuda())
        loss = criterion(output, labels.cuda())
        running_loss += loss.item()
        pred = torch.argmax(torch.exp(output), axis=1)
        pred = pred.detach().cpu().numpy()
        targets = labels.detach().cpu().numpy()
        accuracy = (pred == targets).mean()
        return running_loss, accuracy


def train(epochs):
    steps = 0
    print_every = 10
    for e in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            steps += 1
            optimizer.zero_grad()
            output = model.forward(images.cuda())
            loss = criterion(output, labels.cuda())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if steps % print_every == 0:
                model.eval()
                with torch.no_grad():
                    test_loss, accuracy = validation(test_loader, model)
                print("{}/{} loss {} test_loss {} acc {}".format(e,
                                                                 epochs,
                                                                 running_loss,
                                                                 test_loss,
                                                                 accuracy))
                model.train()
            running_loss = 0.0
        torch.save(model.state_dict(), 'model/_strek_model_save2.pt')
    return model, output


model, output = train(epochs=70)


def piltoarray(img):
    pil_image = transforms.ToPILImage()(img).convert("RGB")
    return np.array(pil_image)


def displayTensorData(x, y):
    train_images = []
    for i in range(x_train.shape[0]):
        train_images.append(piltoarray(x[i]))
    train_images = np.array(train_images)
    display_images(train_images, y.detach().numpy())


def Predict(image):
    model.eval()
    with torch.no_grad():
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        img = cv2.resize(img, (224, 224))
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean, std)(img)
        img = img.reshape(1, 3, 224, 224)
        pred = model(img.cuda())
        pred = torch.argmax(torch.exp(pred), axis=1)
        print(pred)
