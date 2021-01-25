from collections import OrderedDict
from torch import nn
import torch
from torchvision import models

# loading in a vgg-16 pretrained model


def vgg16_pretrain():
    """ This function builds out pretrained classifier model.
    Args:
        None
    Returns: VGG-16 model
    """
    vgg16_model = models.vgg16(pretrained=True)

    for param in vgg16_model.parameters():
        param.requires_grad = False
    # creating a classifier dictionary
    classifer_dict = OrderedDict([("fc1", nn.Linear(25088, 128)),
                                  ("Relu", nn.ReLU(inplace=True)),
                                  ("Dropout1", nn.Dropout(0.5)),
                                  ("fc2", nn.Linear(128, 256)),
                                  ("Relu", nn.ReLU(inplace=True)),
                                  ("Dropout2", nn.Dropout(0.25)),
                                  ("f3", nn.Linear(256, 17)),
                                  ("Softmax", nn.LogSoftmax(dim=1))])
    classifier = nn.Sequential(classifer_dict)
    # replacing the classific method with our own.
    vgg16_model.classifier = classifier.cuda()
    # finding out if gpu is availible if not use cpu
    if torch.cuda.is_available():
        vgg16_model = vgg16_model.cuda()
    else:
        vgg16_model = vgg16_model.cpu()
    return vgg16_model
