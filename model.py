from collections import OrderedDict
from torch import nn
import torch
from torchvision import models

# loading in a vgg-16 pretrained model

# Use Metal (MPS) if available
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
    print("✅ Using Metal (MPS) backend.")
else:
    device = torch.device("cpu")
    print("⚠️ MPS not available. Using CPU.")

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
    vgg16_model.classifier = classifier.to(device)
    # finding out if gpu is availible if not use cpu
    return vgg16_model.to(device)
