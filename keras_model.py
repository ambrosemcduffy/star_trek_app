import pickle as pkl
import numpy as np
from tensorflow.keras import utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Convolution2D, MaxPooling2D


with open("data/train.pkl", "rb") as f:
    X_train, y_train = pkl.load(f)

bottleneck_features = np.load("data//bottleneck_feats.npy")
y_train = np.argmax(y_train, axis=1)
X_train = X_train.astype("float32")/255.
# fig = plt.figure(figsize=(20, 20))
# for i in range(6):
#     ax = fig.add_subplot(1, 6, i+1)
#     ax.imshow(X_train[i], cmap="gray")
#     ax.set_title(str(y_train[i]))


model = Sequential()
model.add(Convolution2D(32, 2,
                        padding="same",
                        activation="relu",
                        input_shape=(224, 224, 3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Convolution2D(64, 2, padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Convolution2D(256, 2, padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten(input_shape=X_train.shape[1:]))
model.add(Dense(512, activation='relu'))
model.add(Dropout(.25))
model.add(Dense(512, activation='relu'))
model.add(Dropout(.25))
model.add(Dense(58, activation="softmax"))
model.summary()

model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metric=["accuracy"])

#y_test = y_binary = to_categorical(y_test)
y_train = y_binary = utils.to_categorical(y_train, 58)

checkpointer = ModelCheckpoint(filepath='data/mnist.model.best.hdf5',
                               verbose=1,
                               save_best_only=True)
hist = model.fit(X_train,
                 y_train,
                 batch_size=128,
                 epochs=90,
                 validation_split=0.1,
                 verbose=1,
                 callbacks=[checkpointer],
                 shuffle=True)

model.load_weights('data/mnist.model.best.hdf5')
score = model.evaluate(X_train, y_train, verbose=0)
accurracy = 100*score
print(accurracy)