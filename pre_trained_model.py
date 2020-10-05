import pickle as pkl
import numpy as np
from tensorflow.keras import utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input


def extract_VGG16(tensor):
    return VGG16(weights='imagenet',
                 include_top=False).predict(preprocess_input(tensor))


with open("data/train.pkl", "rb") as f:
    X_train, y_train = pkl.load(f)

X_train = X_train.astype("float32")/255.

bottleneck_train = extract_VGG16(X_train)


model = Sequential()

model.add(Flatten(input_shape=(7, 7, 512)))
model.add(Dense(512, activation='relu'))
model.add(Dropout(.25))
model.add(Dense(512, activation='relu'))
model.add(Dropout(.25))
model.add(Dense(2, activation="softmax"))
model.summary()

model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metric=["accuracy"])


checkpointer = ModelCheckpoint(filepath='data/mnist.model.best.hdf5',
                               verbose=1,
                               save_best_only=True)
hist = model.fit(bottleneck_train,
                 y_train,
                 batch_size=32,
                 epochs=300,
                 validation_split=0.1,
                 verbose=1,
                 callbacks=[checkpointer],
                 shuffle=True)

model.load_weights('data/mnist.model.best.hdf5')
score = model.evaluate(bottleneck_train, y_train, verbose=0)
accurracy = 100*score
print(accurracy)
