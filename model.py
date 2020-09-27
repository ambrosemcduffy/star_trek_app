import numpy as np
import torch
from torch import nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import cv2

transforms.Compose([transforms.ToTensor(),
                    transforms.Normalize(0.5, 0.5)])


