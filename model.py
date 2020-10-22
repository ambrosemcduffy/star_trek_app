from collections import OrderedDict
from torch import nn
from torchvision import models


def vgg16_pretrain():
    vgg16_model = models.vgg16(pretrained=True)

    for param in vgg16_model.parameters():
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

    vgg16_model.classifier = classifier.cuda()
    vgg16_model = vgg16_model.cuda()
    return vgg16_model
