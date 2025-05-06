import torch
from torch import nn, optim
from collections import OrderedDict
from torchvision import transforms, datasets, models
from gather_data import display_images
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Setting Normalization parameters
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


from PIL import Image
import os

def clean_folder(folder):
    for root, _, files in os.walk(folder):
        for file in files:
            path = os.path.join(root, file)
            try:
                img = Image.open(path)
                img.verify()  # Validate image file
            except (IOError, SyntaxError, Image.UnidentifiedImageError):
                print(f"❌ Removing corrupted image: {path}")
                os.remove(path)

clean_folder("train_set/")
clean_folder("validation_set/")

# Use Metal (MPS) if available
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
    print("✅ Using Metal (MPS) backend.")
else:
    device = torch.device("cpu")
    print("⚠️ MPS not available. Using CPU.")


# Transforming the data

transform = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomRotation([90, 180]),
                                transforms.RandomVerticalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=mean, std=std)])

# Import in the training data
train_data = datasets.ImageFolder("train_set/", transform=transform)
train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=32,
                                           shuffle=True, num_workers=0)
# Importing in the validation data
test_data = datasets.ImageFolder("validation_set/", transform=transform)
test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=32,
                                          shuffle=True, num_workers=0)

model = models.vgg16(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

# Creating a classifier model
classifer_dict = OrderedDict([("fc1", nn.Linear(25088, 128)),
                              ("Relu", nn.ReLU(inplace=True)),
                              ("Dropout1", nn.Dropout(0.5)),
                              ("fc2", nn.Linear(128, 256)),
                              ("Relu", nn.ReLU(inplace=True)),
                              ("Dropout2", nn.Dropout(0.25)),
                              ("f3", nn.Linear(256, 17)),
                              ("Softmax", nn.LogSoftmax(dim=1))])
classifier = nn.Sequential(classifer_dict)

# Checking to see if gpu is availible if not use cpu.
model.classifier = classifier.to(device)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.0007)
criterion = nn.NLLLoss()


def validation(test_loader, model):
    """
    This function is used to see how well we're doing on validation data.
    Args:
        test_loader: Dataloader for validation data.
        model: Model classifer.
    Returns:
        returns training loss, and accuracy.
    """
    running_loss = 0.0
    for images, labels in test_loader:
        output = model.forward(images.to(device))
        loss = criterion(output, labels.to(device))
        running_loss += loss.item()
        pred = torch.argmax(torch.exp(output), axis=1)
        pred = pred.detach().cpu().numpy()
        targets = labels.detach().cpu().numpy()
        accuracy = (pred == targets).mean()
        return running_loss, accuracy


# torch.backends.cudn.benchmark = True


def train(epochs):
    """ This function is used to train the neural network
    Args:
        epochs: number of iterations.
    Returns:
        function returns the model used, and the predictions/outputs
    """
    steps = 0
    print_every = 10
    for e in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            steps += 1
            optimizer.zero_grad()
            output = model.forward(images.to(device))
            loss = criterion(output, labels.to(device))
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


model, output = train(epochs=50)


def piltoarray(img):
    """ 
    This function converts a pil image to a numpy array
    Args:
        img: images
    Returns:
        numpy array
    """
    pil_image = transforms.ToPILImage()(img).convert("RGB")
    return np.array(pil_image)


def displayTensorData(x, y):
    """
    This function helps convert pil images to arrays, and display them.
    Args:
        x: image data
        y: target data/label
    Returns:
        None
    """
    train_images = []
    for i in range(x_train.shape[0]):
        train_images.append(piltoarray(x[i]))
    train_images = np.array(train_images)
    display_images(train_images, y.detach().numpy())
    return None


def Predict(image):
    """
    This function predicts what photo looks like which star trek character
    Args:
        image: photos
    Returns None
    """
    model.eval()
    with torch.no_grad():
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        img = cv2.resize(img, (224, 224))
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean, std)(img)
        img = img.reshape(1, 3, 224, 224)
        pred = model(img.to(device))
        pred = torch.argmax(torch.exp(pred), axis=1)
        print(pred)
    return None
